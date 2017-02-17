#include <deal.II/numerics/vector_tools.h>

#ifndef MODEL_MANAGER_HPP
#define MODEL_MANAGER_HPP

namespace nargil
{

// Forward decleration of model
template <typename ModelEq, int dim, int spacedim> struct model;

// Forward decleration of implicit solver
struct base_implicit_solver;

/**
 *
 *
 * The base class for all other model managers.
 *
 *
 */
template <int dim, int spacedim> struct base_model_manager
{
  /**
   *
   * @brief The deal.II cell iterator type.
   *
   */
  typedef typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator
    dealiiDoFCell;

  /**
   *
   * @brief base_model_manager
   *
   */
  base_model_manager();
};

/**
 *
 *
 * The main class for all models containing hybridized cells.
 *
 *
 */
template <int dim, int spacedim = dim>
struct implicit_hybridized_model_manager : base_model_manager<dim, spacedim>
{
  /**
   *
   * The dealii cell iterator containing dof data.
   *
   */
  using typename base_model_manager<dim, spacedim>::dealiiDoFCell;

  /**
   *
   * @brief hybridized_model_manager
   *
   */
  implicit_hybridized_model_manager();

  /**
   *
   * Here we interpolate a given function on to the space od local dofs
   *
   */
  //  template <typename VectorType> VectorType interpolate_func_to_dof() {}

  /**
   *
   * Here we form the dof_handler of the manager.
   *
   */
  template <typename BasisType, typename ModelEq>
  void form_dof_handlers(model<ModelEq, dim, spacedim> *in_model,
                         BasisType *in_basis);

  /**
   *
   * This function invokes the function f with the arguments args
   * as the arguments of f for each active element of the mesh.
   *
   */
  template <typename ModelEq, typename Func, typename... Args>
  void apply_func_to_owned_cells(model<ModelEq, dim, spacedim> *in_model,
                                 Func f,
                                 Args... args);

  /**
   *
   * dof handler containing the local dof data of cells.
   *
   */
  dealii::DoFHandler<dim, spacedim> local_dof_handler;

  /**
   *
   * dof handler containing the trace dof of the cells.
   *
   */
  dealii::DoFHandler<dim, spacedim> trace_dof_handler;
};
}

#include "../../source/models/model_manager.cpp"

#endif
