#include "../../include/elements/cell.hpp"

template <int dim, int spacedim>
nargil::cell_operators<dim, spacedim>::cell_operators(
  nargil::cell<dim, spacedim> *const in_cell)
  : my_cell(in_cell)
{
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
