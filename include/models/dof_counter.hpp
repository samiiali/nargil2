#ifndef IMPLICIT_DOF_NUM_HPP
#define IMPLICIT_DOF_NUM_HPP

#include <map>
#include <memory>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include <mpi.h>

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/tria.h>

#include "../../include/elements/cell.hpp"
#include "../../include/misc/utils.hpp"

namespace nargil
{

// Forward declerations
template <typename ModelEq, int dim, int spacedim> struct model;

/**
 *
 * The base class for numbering the degrees of freedom of the model. It is
 * not supposed to be used by the user. It will be contained inside the Model
 * class. It might seem ok to include a pointer to the containing
 * nargil::base_model in the dof_counter. But it is totally not required.
 * Because the user wil interact with model object and not the dof_counter
 * object. So, when the user calls a method in nargil::model, we decide which
 * dof_counter to call based on the stored key of the dof_counter.
 *
 */
template <int dim, int spacedim = dim> struct dof_counter
{
  /**
   * @brief The deal.II cell iterator type.
   */

  /**
   *
   * @brief The deal.II cell iterator type.
   *
   */
  typedef typename dealii::Triangulation<dim, spacedim>::active_cell_iterator
    dealiiTriCell;

  /**
   *
   * @brief Constructor of the class.
   *
   */
  dof_counter();

  /**
   *
   * @brief Destructor of the class.
   *
   */
  virtual ~dof_counter();

  /**
   *
   * @brief Number of DOFs which are owned by this CPU.
   *
   */
  unsigned n_global_unkns_rank_owns;

  /**
   *
   * @brief Total number of DOFs on all of the ranks.
   *
   */
  unsigned n_global_unkns_on_all_ranks;

  /**
   *
   * @brief Total number of DOFs on all of the ranks.
   *
   */
  unsigned n_local_unkns_on_this_rank;

  /**
   *
   * The @c ith member of this std::vector contains the number of
   * DOFs, which are owned by the current rank and are connected to the
   * @c ith DOF.
   *
   */
  std::vector<int> n_local_unkns_connected_to_unkn;

  /**
   *
   * The @c ith member of this std::vector contains the number of
   * DOFs, which are NOT owned by the current rank and are connected to the
   * @c ith DOF.
   *
   */
  std::vector<int> n_nonlocal_unkns_connected_to_unkn;

  /**
   *
   * The global number of each DOF which are present on the current
   * rank (either owned by this rank or not).
   *
   */
  std::vector<int> scatter_from;

  /**
   *
   * The local number of each DOF which are present on the current
   * rank (either owned by this rank or not).
   *
   */
  std::vector<int> scatter_to;
};

/**
 *
 *
 * This class enumerate the unknowns in a model which contains
 * hybridized DG elements.
 *
 *
 */
template <int dim, int spacedim = dim>
struct implicit_hybridized_numbering : public dof_counter<dim, spacedim>
{
  /**
   *
   * dealiiTriCell
   *
   */
  using typename dof_counter<dim, spacedim>::dealiiTriCell;

  /**
   *
   * @brief The constructor of the class.
   *
   */
  implicit_hybridized_numbering();

  /**
   *
   * @brief The destructor of the class.
   *
   */
  ~implicit_hybridized_numbering();

  /**
   *
   * This function counts the unknowns according to the model type and
   * model equation. It is called from model::count_globals().
   *
   * @todo We cast dof_counter::my_model to nargil::model. This can be
   * avoided. We use the same basis for ghost cells as regular cells. This can
   * be avoided as well.
   *
   */
  template <typename BasisType, typename ModelEq>
  void count_globals(model<ModelEq, dim, spacedim> *my_model);
};
}

#include "../../source/models/dof_counter.cpp"

#endif
