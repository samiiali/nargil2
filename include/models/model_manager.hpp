#include <deal.II/numerics/vector_tools.h>

#include "../misc/utils.hpp"

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
 * The main class for all models containing hybridized cells.
 *
 *
 */
template <int dim, int spacedim = dim> struct hybridized_model_manager
{
  /**
   *
   * @brief hybridized_model_manager
   *
   */
  hybridized_model_manager();

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
  void apply_on_owned_cells(model<ModelEq, dim, spacedim> *in_model,
                            Func f,
                            Args... args);

  /**
   *
   * This function invokes the function f with the arguments args
   * as the arguments of f for each ghost element of the mesh.
   *
   */
  template <typename ModelEq, typename Func, typename... Args>
  void apply_on_ghost_cells(model<ModelEq, dim, spacedim> *in_model,
                            Func f,
                            Args... args);

  /**
   *
   * This function connects the elements from another model to the elements
   * of the current model.
   *
   */
  template <typename ModelEq, typename OtherModelEq>
  void
  connect_to_other_model(model<ModelEq, dim, spacedim> *in_model,
                         model<OtherModelEq, dim, spacedim> *in_other_model);

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

  /**
   *
   * The dealii DoFHandler which is used for refining the grid.
   *
   */
  dealii::DoFHandler<dim, spacedim> refn_dof_handler;

  /**
   *
   * The dealii DoFHandler which is used for refining the grid.
   *
   */
  dealii::DoFHandler<dim, spacedim> viz_dof_handler;
};
}

#include "../../source/models/model_manager.cpp"

#endif
