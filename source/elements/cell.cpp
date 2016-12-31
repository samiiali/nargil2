#include "../../include/elements/cell.hpp"

template <int dim, int spacedim>
nargil::bases::hdg_diffusion_polybasis<dim, spacedim>::hdg_diffusion_polybasis(
  const unsigned poly_order, const unsigned quad_order)
  : _poly_order(poly_order),
    _quad_order(quad_order),
    u_basis(poly_order),
    q_basis(u_basis, dim),
    uhat_basis(poly_order),
    cell_quad(quad_order),
    face_quad(quad_order),
    fe_val_flags(dealii::update_values | dealii::update_gradients |
                 dealii::update_JxW_values | dealii::update_quadrature_points),
    u_in_cell(u_basis, cell_quad, fe_val_flags),
    q_in_cell(q_basis, cell_quad, fe_val_flags),
    uhat_on_face(u_basis, face_quad, fe_val_flags),
    u_on_faces(2 * dim),
    q_on_faces(2 * dim)
{
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    dealii::Quadrature<dim> projected_i_face_quad =
      dealii::QProjector<dim>::project_to_face(face_quad, i_face);
    std::unique_ptr<dealii::FEValues<dim> > u_on_i_face(
      new dealii::FEValues<dim>(u_basis, projected_i_face_quad, fe_val_flags));
    std::unique_ptr<dealii::FEValues<dim> > q_on_i_face(
      new dealii::FEValues<dim>(q_basis, projected_i_face_quad, fe_val_flags));
    u_on_faces[i_face] = std::move(u_on_i_face);
    q_on_faces[i_face] = std::move(q_on_i_face);
  }
}

//
//

template <int dim, int spacedim>
void nargil::bases::hdg_diffusion_polybasis<dim, spacedim>::
  adjusted_subface_quad_points(const dealii::Point<dim - 1> &in_point,
                               const unsigned half_range)
{
  assert(half_range <= pow(2, in_point.dimension));
  std::vector<double> result;
  dealii::Point<dim - 1> out_point(in_point);
  if (half_range != 0)
  {
    if (in_point.dimension == 1)
    {
      if (half_range == 1)
        out_point(0) = in_point(0) / 2.0;
      if (half_range == 2)
        out_point(0) = 0.5 + in_point(0) / 2.0;
    }
    if (in_point.dimension == 2)
    {
      if (half_range == 1)
      {
        out_point(0) = in_point(0) / 2.0;
        out_point(1) = in_point(1) / 2.0;
      }
      if (half_range == 2)
      {
        out_point(0) = 0.5 + in_point(0) / 2.0;
        out_point(1) = in_point(1) / 2.0;
      }
      if (half_range == 3)
      {
        out_point(0) = in_point(0) / 2.0;
        out_point(1) = 0.5 + in_point(1) / 2.0;
      }
      if (half_range == 4)
      {
        out_point(0) = 0.5 + in_point(0) / 2.0;
        out_point(1) = 0.5 + in_point(1) / 2.0;
      }
    }
  }
  return out_point;
}

//
//

template <int dim, int spacedim>
unsigned
nargil::bases::hdg_diffusion_polybasis<dim, spacedim>::get_n_dofs_on_each_face()
{
  return uhat_basis.dofs_per_cell;
}

//
//

template <int dim, int spacedim>
nargil::bases::basis_options
nargil::bases::hdg_diffusion_polybasis<dim, spacedim>::get_options()
{
  return (bases::basis_options)(basis_options::HDG | basis_options::nodal |
                                basis_options::polynomial);
}

//
//
//
//
//
template <int dim, int spacedim>
nargil::cell<dim, spacedim>::cell(dealii_cell_type &inp_cell,
                                  const unsigned id_num_,
                                  base_model *model_)
  : n_faces(dealii::GeometryInfo<dim>::faces_per_cell),
    //    poly_order(poly_order_),
    //    n_face_bases(pow(poly_order + 1, dim - 1)),
    //    n_cell_bases(pow(poly_order + 1, dim)),
    id_num(id_num_),
    dof_names_on_faces(n_faces),
    half_range_flag(n_faces, 0),
    face_owner_rank(n_faces, -1),
    dealii_cell(inp_cell),
    dofs_ID_in_this_rank(n_faces),
    dofs_ID_in_all_ranks(n_faces),
    BCs(n_faces, boundary_condition::not_set),
    my_model(model_)
{
  std::stringstream ss_id;
  ss_id << inp_cell->id();
  cell_id = ss_id.str();
}

//
//

template <int dim, int spacedim>
nargil::cell<dim, spacedim>::cell(cell &&inp_cell) noexcept
  : n_faces(std::move(inp_cell.n_faces)),
    //    poly_order(std::move(inp_cell.poly_order)),
    //    n_face_bases(std::move(inp_cell.n_face_bases)),
    //    n_cell_bases(std::move(inp_cell.n_cell_bases)),
    id_num(std::move(inp_cell.id_num)),
    dof_names_on_faces(std::move(inp_cell.dof_names_on_faces)),
    cell_id(std::move(inp_cell.cell_id)),
    half_range_flag(std::move(inp_cell.half_range_flag)),
    face_owner_rank(std::move(inp_cell.face_owner_rank)),
    dealii_cell(std::move(inp_cell.dealii_cell)),
    dofs_ID_in_this_rank(std::move(inp_cell.dofs_ID_in_this_rank)),
    dofs_ID_in_all_ranks(std::move(inp_cell.dofs_ID_in_all_ranks)),
    BCs(std::move(inp_cell.BCs)),
    my_model(inp_cell.my_model)
{
}

//
//

template <int dim, int spacedim> nargil::cell<dim, spacedim>::~cell() {}

//
//

template <int dim, int spacedim>
template <typename ModelEq, typename BasisType>
std::unique_ptr<ModelEq>
nargil::cell<dim, spacedim>::create(dealii_cell_type &inp_cell,
                                    const unsigned id_num_,
                                    BasisType *basis,
                                    base_model *model_)
{
  return std::unique_ptr<ModelEq>(
    new ModelEq(inp_cell, id_num_, basis, model_, BasisType::get_options()));
}

//
//

template <int dim, int spacedim>
void nargil::cell<dim, spacedim>::reinit_cell_fe_vals()
{
  // cell_quad_fe_vals->reinit(dealii_cell);
  // cell_supp_fe_vals->reinit(dealii_cell);
}

//
//

template <int dim, int spacedim>
void nargil::cell<dim, spacedim>::reinit_face_fe_vals(unsigned)
{
  // face_quad_fe_vals->reinit(dealii_cell, i_face);
  // face_supp_fe_vals->reinit(dealii_cell, i_face);
}

//
//

template <int dim, int spacedim>
void nargil::cell<dim, spacedim>::assign_local_global_cell_data(
  const unsigned &i_face,
  const unsigned &local_num_,
  const unsigned &global_num_,
  const unsigned &comm_rank_,
  const unsigned &half_range_)
{
  face_owner_rank[i_face] = comm_rank_;
  half_range_flag[i_face] = half_range_;
  for (unsigned i_dof = 0; i_dof < dof_names_on_faces[i_face].count(); ++i_dof)
  {
    dofs_ID_in_this_rank[i_face].push_back(local_num_ + i_dof);
    dofs_ID_in_all_ranks[i_face].push_back(global_num_ + i_dof);
  }
}

//
//

template <int dim, int spacedim>
void nargil::cell<dim, spacedim>::assign_ghost_cell_data(
  const unsigned &i_face,
  const int &local_num_,
  const int &global_num_,
  const unsigned &comm_rank_,
  const unsigned &half_range_)
{
  face_owner_rank[i_face] = comm_rank_;
  half_range_flag[i_face] = half_range_;
  for (unsigned i_dof = 0; i_dof < dof_names_on_faces[i_face].count(); ++i_dof)
  {
    dofs_ID_in_this_rank[i_face].push_back(local_num_ - i_dof);
    dofs_ID_in_all_ranks[i_face].push_back(global_num_ - i_dof);
  }
}

//
//

template <int dim, int spacedim>
void nargil::cell<dim, spacedim>::assign_local_cell_data(
  const unsigned &i_face,
  const unsigned &local_num_,
  const int &comm_rank_,
  const unsigned &half_range_)
{
  face_owner_rank[i_face] = comm_rank_;
  half_range_flag[i_face] = half_range_;
  for (unsigned i_dof = 0; i_dof < dof_names_on_faces[i_face].count(); ++i_dof)
    dofs_ID_in_this_rank[i_face].push_back(local_num_ + i_dof);
}

//
//
//
//
//
template <int dim, int spacedim>
nargil::diffusion_cell<dim, spacedim>::diffusion_cell(
  dealii_cell_type &inp_cell,
  const unsigned id_num_,
  bases::basis<dim, spacedim> *basis,
  base_model *model_,
  bases::basis_options basis_opts)
  : cell<dim, spacedim>(inp_cell, id_num_, model_),
    my_basis(basis),
    my_basis_opts(basis_opts)
{
  std::cout << "Constructor of diffusion cell" << std::endl;
}

//
//

template <int dim, int spacedim>
template <typename Func>
void nargil::diffusion_cell<dim, spacedim>::assign_BCs(Func f)
{
  std::cout << "assign_BC at diffusion_cell" << std::endl;
}

//
//

template <int dim, int spacedim>
unsigned nargil::diffusion_cell<dim, spacedim>::get_relevant_dofs_count(
  const unsigned i_face)
{
  unsigned num_dofs_on_face =
    static_cast<bases::hdg_diffusion_polybasis<dim, spacedim> *>(my_basis)
      ->get_n_dofs_on_each_face();
  std::cout << i_face << " " << num_dofs_on_face << std::endl;
  return num_dofs_on_face;
}

//
//

template <int dim, int spacedim>
void nargil::diffusion_cell<dim, spacedim>::assemble_globals()
{
}
