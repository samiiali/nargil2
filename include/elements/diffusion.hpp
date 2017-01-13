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
/**
 * @brief The element to be used for solving the diffusion equation.
 *
 * The original method of solving this is
 * based on hybridized DG.
 * \ingroup modelelements
 */
template <int dim, int spacedim = dim>
struct diffusion : public cell<dim, spacedim>
{
  //
  //
  /**
   * @brief We use the same typename as we defined in the base class (cell).
   */
  using typename cell<dim, spacedim>::dealii_cell_type;

  //
  //
  /**
   * @brief The constructor of the class.
   */
  diffusion(dealii_cell_type &in_cell,
            const unsigned id_num_,
            const base_basis<dim, spacedim> *base_basis,
            base_model *in_model);

  //
  //
  /**
   * @brief The destructor of the class.
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
   * @brief Assigns the boundary condition to different DOFs.
   */
  template <typename BasisType, typename Func> void assign_BCs(Func f);

  //
  //
  /**
   *
   */
  template <typename CellManagerType> CellManagerType *get_manager();

  //
  //
  /**
   *
   */
  template <typename BasisType> const BasisType *get_basis() const;

  //
  //
  /**
   * Polynomial basis of type HDG.
   * @ingroup modelbases
   */
  struct hdg_polybasis : public base_basis<dim, spacedim>
  {
    //
    //
    /**
     * @brief relevant_manager_type
     */
    typedef typename diffusion::hdg_manager relevant_manager_type;

    //
    /**
     * @brief hdg_polybasis
     */
    hdg_polybasis() = delete;

    //
    //
    /**
     *
     */
    virtual ~hdg_polybasis() final;

    //
    /**
     * @brief Class constructor
     */
    hdg_polybasis(const unsigned poly_order, const unsigned quad_order);

    //
    /**
     * @brief _poly_order
     */
    unsigned _poly_order;

    //
    /**
     * @brief _quad_order
     */
    unsigned _quad_order;

    //
    /**
     * @brief u_basis
     */
    dealii::FE_DGQ<dim> u_basis;

    //
    /**
     * @brief q_basis
     */
    dealii::FESystem<dim> q_basis;

    //
    /**
     * @brief uhat_basis
     */
    dealii::FE_DGQ<dim - 1> uhat_basis;

    //
    /**
     * @brief cell_quad
     */
    dealii::QGauss<dim> cell_quad;

    //
    /**
     * @brief face_quad
     */
    dealii::QGauss<dim - 1> face_quad;

    //
    /**
     * @brief projected_face_quad
     */
    dealii::Quadrature<dim> projected_face_quad;

    //
    /**
     * @brief fe_val_flags
     */
    dealii::UpdateFlags fe_val_flags;

    //
    /**
     * @brief u_in_cell
     */
    dealii::FEValues<dim> u_in_cell;

    //
    /**
     * @brief q_in_cell
     */
    dealii::FEValues<dim> q_in_cell;

    //
    /**
     * @brief uhat_on_face
     */
    dealii::FEFaceValues<dim> uhat_on_face;

    //
    /**
     * @brief u_on_faces
     */
    std::vector<std::unique_ptr<dealii::FEValues<dim> > > u_on_faces;

    //
    /**
     * @brief q_on_faces
     */
    std::vector<std::unique_ptr<dealii::FEValues<dim> > > q_on_faces;

    //
    //
    /**
     * @brief get_n_dofs_on_each_face
     */
    unsigned get_n_dofs_on_each_face() const;

    //
    //
    /**
     * @brief adjusted_subface_quad_points
     */
    void adjusted_subface_quad_points(const dealii::Point<dim - 1> &P0,
                                      const unsigned half_range);

    //
    //
    /**
     *
     */
    unsigned n_unknowns_for_ith_dof(const unsigned i_dof) const;
  };

  //
  //
  /**
   * @brief The hdg_operations struct
   */
  struct hdg_manager : hybridized_cell_manager<dim, spacedim>
  {
    /**
     * @brief hdg_manager
     */
    hdg_manager(const diffusion<dim, spacedim> *);

    //
    //
    /**
     * @brief Deconstructor of the class
     */
    virtual ~hdg_manager() final;

    //
    //
    /**
     *
     */
    template <typename Func> void assign_BCs(Func f);
  };

  //
  //
  /**
   * @brief my_basis
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
};
}

#include "../../source/elements/diffusion.cpp"

#endif
