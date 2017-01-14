#include "../../include/elements/diffusion.hpp"

//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::diffusion(
  dealii_cell_type &inp_cell,
  const unsigned in_id_num,
  const base_basis<dim, spacedim> *in_basis,
  base_model *model_)
  : cell<dim, spacedim>(inp_cell, in_id_num, model_), my_basis(in_basis)
{
}

//
//

template <int dim, int spacedim> nargil::diffusion<dim, spacedim>::~diffusion()
{
}

//
//

template <int dim, int spacedim>
template <typename CellManagerType>
void nargil::diffusion<dim, spacedim>::diffusion::init_manager()
{
  my_manager =
    std::move(std::unique_ptr<CellManagerType>(new CellManagerType(this)));
}

//
//

template <int dim, int spacedim>
template <typename BasisType, typename Func>
void nargil::diffusion<dim, spacedim>::assign_BCs(Func f)
{
  static_cast<typename BasisType::relevant_manager_type *>(my_manager.get())
    ->assign_BCs(f);
}

//
//

template <int dim, int spacedim>
template <typename CellManagerType>
CellManagerType *nargil::diffusion<dim, spacedim>::get_manager()
{
  return static_cast<CellManagerType *>(my_manager.get());
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
const BasisType *nargil::diffusion<dim, spacedim>::get_basis() const
{
  return static_cast<const BasisType *>(my_basis);
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::hdg_polybasis::hdg_polybasis(
  const unsigned poly_order, const unsigned quad_order)
  : base_basis<dim, spacedim>(),
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
nargil::diffusion<dim, spacedim>::hdg_polybasis::~hdg_polybasis()
{
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
unsigned nargil::diffusion<dim, spacedim>::hdg_polybasis::get_n_dofs_per_face()
{
  return 1;
}

//
//

template <int dim, int spacedim>
std::vector<unsigned>
nargil::diffusion<dim, spacedim>::hdg_polybasis::get_n_unkns_per_dofs() const
{
  // Here, we will have copy elision, DO NOT try to optimize using move
  // semantics.
  std::vector<unsigned> n_unkns_per_dofs(1, uhat_basis.dofs_per_cell);
  return n_unkns_per_dofs;
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::hdg_manager::hdg_manager(
  const nargil::diffusion<dim, spacedim> *in_cell)
  : hybridized_cell_manager<dim, spacedim>(in_cell)
{
}

//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::hdg_manager::~hdg_manager()
{
}

//
//

template <int dim, int spacedim>
template <typename Func>
void nargil::diffusion<dim, spacedim>::hdg_manager::assign_BCs(Func f)
{
  f(this);
}
