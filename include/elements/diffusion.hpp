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
  diffusion(dealii_cell_type &inp_cell,
            const unsigned id_num_,
            basis<dim, spacedim> *basis,
            base_model *model_,
            bases_options::options basis_opts);

  //
  //
  /**
   * @brief The destructor of the class.
   */
  ~diffusion() {}

  //
  //
  /**
   * @brief Assigns the boundary condition to different DOFs.
   */
  template <typename Func> void assign_BCs(Func f);

  //
  //
  /**
   * @brief Assembles the global matrices and vectors for solving the
   * problem.
   */
  void assemble_globals();

  //
  //
  /**
   * @brief Gets the open dofs corresponding to the current basis.
   */
  unsigned get_relevant_dofs_count(const unsigned);

  //
  //
  /**
   * @brief Options for the basis.
   */
  bases_options::options my_basis_opts;

  //
  //
  /**
   * Polynomial basis of type HDG.
   * @ingroup modelbases
   */
  struct hdg_polybasis : public basis<dim, spacedim>
  {
    //
    /**
     * @brief hdg_polybasis
     */
    hdg_polybasis() = delete;

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
    unsigned get_n_dofs_on_each_face();

    //
    //
    /**
     * @brief get_options
     */
    static bases_options::options get_options();

    //
    //
    /**
     * @brief adjusted_subface_quad_points
     */
    void adjusted_subface_quad_points(const dealii::Point<dim - 1> &P0,
                                      const unsigned half_range);
  };

  //
  //
  /**
   * @brief The hdg_operations struct
   */
  struct hdg_operators : cell_operators<dim, spacedim>
  {
    /**
     * @brief hdg_operators
     */
    hdg_operators(cell<dim, spacedim> *);

    //
    //
    /**
     * @brief get_options
     */
    static int get_options();

    //
    //
    /**
     * @brief Deconstructor of the class
     */
    ~hdg_operators() {}

    //
    //
    /**
     * We want to know which degrees of freedom are restrained and which are
     * open. Hence, we store a bitset which has its size equal to the number of
     * dofs of each face of the cell and it is 1 if the dof is open, and 0 if it
     * is restrained.
     */
    std::vector<boost::dynamic_bitset<> > dof_names_on_faces;
  };

  //
  //
  /**
   * @brief my_basis
   */
  basis<dim, spacedim> *my_basis;

  //
  //
  /**
   * @brief my_operators
   */
  std::unique_ptr<cell_operators<dim, spacedim> > my_operators;
};
}

#include "../../source/elements/diffusion.cpp"

#endif
