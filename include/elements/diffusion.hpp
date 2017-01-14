#include <type_traits>

#include <boost/dynamic_bitset.hpp>

#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include "../models/model_options.hpp"
#include "cell.hpp"

#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP

namespace nargil
{
//
//
// Forward decleration of base_model, to be used in diffusion.
struct base_model;

//
//
//
//
/**
 * The element to be used for solving the diffusion equation.
 * The first method of solving this is based on hybridized DG.
 *
 * \ingroup modelelements
 */
template <int dim, int spacedim = dim>
struct diffusion : public cell<dim, spacedim>
{
  //
  //
  /**
   * We use the same typename as we defined in the base class
   * (nargil::cell).
   */
  using typename cell<dim, spacedim>::dealii_cell_type;

  //
  //
  /**
   * The constructor of the class.
   */
  diffusion(dealii_cell_type &in_cell,
            const unsigned id_num_,
            const base_basis<dim, spacedim> *base_basis,
            base_model *in_model);

  //
  //
  /**
   * The destructor of the class.
   */
  virtual ~diffusion() final;

  //
  //
  /**
   * This function is called from cell::create. This function cannot
   * be const, because the diffusion::my_manager is changed in this
   * function.
   */
  template <typename CellManagerType> void init_manager();

  //
  //
  /**
   *  Assigns the boundary condition to different DOFs.
   */
  template <typename BasisType, typename Func> void assign_BCs(Func f);

  //
  //
  /**
   * This function static_cast the manager of this cell to an appropriate
   * type and return it to the user.
   */
  template <typename CellManagerType> CellManagerType *get_manager();

  //
  //
  /**
   * This function static_cast the basis of this cell to an appropriate
   * type and return it to the user.
   */
  template <typename BasisType> const BasisType *get_basis() const;

  //
  //
  /**
   * Contains the basis of the current cell.
   */
  const base_basis<dim, spacedim> *my_basis;

  //
  //
  /**
   * The manager of this element. The reason to define this manager
   * as a unique_ptr is we need its polymorphic features in the code.
   * Hence, we are able to use static_cast to cast this manager to derived
   * types of cell_manager.
   */
  std::unique_ptr<cell_manager<dim, spacedim> > my_manager;

  //
  //
  //
  //
  /**
   * Polynomial basis of type HDG.
   *
   * @ingroup modelbases
   */
  struct hdg_polybasis : public base_basis<dim, spacedim>
  {
    //
    //
    /**
     *  relevant_manager_type
     */
    typedef typename diffusion::hdg_manager relevant_manager_type;

    //
    //
    /**
     *  hdg_polybasis
     */
    hdg_polybasis() = delete;

    //
    //
    /**
     *  Class constructor
     */
    hdg_polybasis(const unsigned poly_order, const unsigned quad_order);

    //
    //
    /**
     *
     */
    virtual ~hdg_polybasis() final;

    //
    //
    /**
     * This function returns the number of dofs on each face of the element.
     * The term dof refers to the functions which we want to solve a global
     * equations to obtain them. For example in diffusion equation we solve
     * for u globally. So, the only global dof is u, and this function
     * returns 1.
     */
    static unsigned get_n_dofs_per_face();

    //
    //
    /**
     *  get_n_unkns_per_dofs
     */
    std::vector<unsigned> get_n_unkns_per_dofs() const;

    //
    //
    /**
     *  adjusted_subface_quad_points
     */
    void adjusted_subface_quad_points(const dealii::Point<dim - 1> &P0,
                                      const unsigned half_range);

    //
    //
    /**
     *  _poly_order
     */
    unsigned _poly_order;

    //
    //
    /**
     *  _quad_order
     */
    unsigned _quad_order;

    //
    //
    /**
     *  u_basis
     */
    dealii::FE_DGQ<dim> u_basis;

    //
    //
    /**
     *  q_basis
     */
    dealii::FESystem<dim> q_basis;

    //
    //
    /**
     *  uhat_basis
     */
    dealii::FE_DGQ<dim - 1> uhat_basis;

    //
    //
    /**
     *  cell_quad
     */
    dealii::QGauss<dim> cell_quad;

    //
    //
    /**
     *  face_quad
     */
    dealii::QGauss<dim - 1> face_quad;

    //
    //
    /**
     *  projected_face_quad
     */
    dealii::Quadrature<dim> projected_face_quad;

    //
    //
    /**
     *  fe_val_flags
     */
    dealii::UpdateFlags fe_val_flags;

    //
    //
    /**
     *  u_in_cell
     */
    dealii::FEValues<dim> u_in_cell;

    //
    //
    /**
     *  q_in_cell
     */
    dealii::FEValues<dim> q_in_cell;

    //
    //
    /**
     *  uhat_on_face
     */
    dealii::FEFaceValues<dim> uhat_on_face;

    //
    //
    /**
     *  u_on_faces
     */
    std::vector<std::unique_ptr<dealii::FEValues<dim> > > u_on_faces;

    //
    //
    /**
     *  q_on_faces
     */
    std::vector<std::unique_ptr<dealii::FEValues<dim> > > q_on_faces;
  };

  //
  //
  //
  //
  /**
   *  The hdg_manager struct
   */
  struct hdg_manager : hybridized_cell_manager<dim, spacedim>
  {
    /**
     *  hdg_manager
     */
    hdg_manager(const diffusion<dim, spacedim> *);

    //
    //
    /**
     *  Deconstructor of the class
     */
    virtual ~hdg_manager() final;

    //
    //
    /**
     * Sets the boundary conditions of the cell.
     */
    template <typename Func> void assign_BCs(Func f);
  };
};
}

#include "../../source/elements/diffusion.cpp"

#endif
