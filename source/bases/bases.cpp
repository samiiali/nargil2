#include "../../include/bases/bases.hpp"

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
