#include "../../include/elements/diffusion.hpp"

//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::diffusion(dealii_cell_type &inp_cell,
                                            const unsigned id_num_,
                                            basis<dim, spacedim> *basis,
                                            base_model *model_,
                                            bases_options::options basis_opts)
  : cell<dim, spacedim>(inp_cell, id_num_, model_),
    my_basis_opts(basis_opts),
    my_basis(basis)
{
  if (my_basis_opts & hdg_worker::get_options())
  {
    my_worker = std::move(std::unique_ptr<hdg_worker>(new hdg_worker(this)));
  }
  else
  {
    std::cout << "Options in diffusion class constructor were not recognoized."
              << std::endl;
  }
  std::cout << "Constructor of diffusion cell" << std::endl;
}

//
//

template <int dim, int spacedim>
template <typename Func>
void nargil::diffusion<dim, spacedim>::assign_BCs(Func f)
{
  if (my_basis_opts & hdg_worker::get_options())
  {
    static_cast<hdg_worker *>(my_worker.get())->assign_BCs(f);
  }
  else
  {
    std::cout << "Options in diffusion assign_BCs were not recognized."
              << std::endl;
  }
}

//
//

template <int dim, int spacedim>
unsigned
nargil::diffusion<dim, spacedim>::get_relevant_dofs_count(const unsigned i_face)
{
  unsigned num_dofs_on_face =
    static_cast<hdg_polybasis *>(my_basis)->get_n_dofs_on_each_face();
  std::cout << i_face << " " << num_dofs_on_face << std::endl;
  return num_dofs_on_face;
}

//
//

template <int dim, int spacedim>
void nargil::diffusion<dim, spacedim>::assemble_globals()
{
}

//
//

template <int dim, int spacedim>
nargil::cell_worker<dim, spacedim> *
nargil::diffusion<dim, spacedim>::get_worker()
{
  //  if (my_basis_opts & hdg_worker::get_options())
  //  {
  return static_cast<hdg_worker *>(my_worker.get());
  //  }
  //  else
  //  {
  //    std::cout << "Options in diffusion get_worker were not recognized."
  //              << std::endl;
  //  }
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::hdg_polybasis::hdg_polybasis(
  const unsigned poly_order, const unsigned quad_order)
  : basis<dim, spacedim>(),
    _poly_order(poly_order),
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
void nargil::diffusion<dim, spacedim>::hdg_polybasis::
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
nargil::diffusion<dim, spacedim>::hdg_polybasis::get_n_dofs_on_each_face()
{
  return uhat_basis.dofs_per_cell;
}

//
//

template <int dim, int spacedim>
nargil::bases_options::options
nargil::diffusion<dim, spacedim>::hdg_polybasis::get_options()
{
  return (bases_options::options)(bases_options::HDG | bases_options::nodal |
                                  bases_options::polynomial);
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::hdg_worker::hdg_worker(
  nargil::cell<dim, spacedim> *in_cell)
  : cell_worker<dim, spacedim>(in_cell),
    dofs_ID_in_this_rank(2 * dim),
    dofs_ID_in_all_ranks(2 * dim),
    BCs(2 * dim, boundary_condition::not_set),
    dof_names_on_faces(2 * dim)
// 2 * dim is actually the number of element faces.
{
}

//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::hdg_worker::~hdg_worker()
{
}

//
//

template <int dim, int spacedim>
int nargil::diffusion<dim, spacedim>::hdg_worker::get_options()
{
  return (bases_options::HDG);
}

//
//

template <int dim, int spacedim>
template <typename Func>
void nargil::diffusion<dim, spacedim>::hdg_worker::assign_BCs(Func f)
{
  f(this);
}

//
//
