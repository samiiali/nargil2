#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#ifndef BASES_HPP
#define BASES_HPP

namespace nargil
{

//
//

namespace bases
{

//
//
/**
 * @brief The BasesOptions enum
 */
enum basis_options
{
  HDG = 1 << 0,
  nodal = 1 << 3,
  polynomial = 1 << 5
};
//
//
/**
 *
 */
template <int dim, int spacedim> struct basis
{
  basis() {}
  ~basis() {}
  template <typename BasisType> std::unique_ptr<BasisType> create();
};

//
//
/**
 *
 */
template <int dim, int spacedim = dim>
struct hdg_diffusion_polybasis : public basis<dim, spacedim>
{
  hdg_diffusion_polybasis() = delete;
  hdg_diffusion_polybasis(const unsigned poly_order, const unsigned quad_order);

  unsigned _poly_order;
  unsigned _quad_order;

  dealii::FE_DGQ<dim> u_basis;
  dealii::FESystem<dim> q_basis;
  dealii::FE_DGQ<dim - 1> uhat_basis;

  dealii::QGauss<dim> cell_quad;
  dealii::QGauss<dim - 1> face_quad;

  dealii::Quadrature<dim> projected_face_quad;

  dealii::UpdateFlags fe_val_flags;

  dealii::FEValues<dim> u_in_cell;
  dealii::FEValues<dim> q_in_cell;
  dealii::FEFaceValues<dim> uhat_on_face;
  std::vector<std::unique_ptr<dealii::FEValues<dim> > > u_on_faces;
  std::vector<std::unique_ptr<dealii::FEValues<dim> > > q_on_faces;

  //
  //
  /**
   *
   */
  void adjusted_subface_quad_points(const dealii::Point<dim - 1> &P0,
                                    const unsigned half_range);

  //
  //
  /**
   *
   */
  unsigned get_n_dofs_on_each_face();

  //
  //
  /**
   *
   */
  static nargil::bases::basis_options get_options();
};
}
}

#include "../../source/bases/bases.cpp"

#endif
