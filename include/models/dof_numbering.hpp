#ifndef IMPLICIT_DOF_NUM_HPP
#define IMPLICIT_DOF_NUM_HPP

#include "model_options.hpp"

namespace nargil
{
//
//
/**
 * The base class for numbering the degrees of freedom of the model. It is
 * not supposed to be used by the user. It will be contained inside the Model
 * class.
 */
template <int dim, int spacedim = dim> struct dof_numbering
{
  //
  //
  /**
   * @brief Constructor of the class.
   */
  dof_numbering();

  //
  //
  /**
   * @brief Destructor of the class.
   */
  virtual ~dof_numbering();

  //
  //
  /**
   * @brief Rank number of the CPU containing the Model.
   */
  unsigned comm_rank;

  //
  //
  /**
   * @brief Number of DOFs which are owned by this CPU.
   */
  unsigned n_global_DOFs_rank_owns;

  //
  //
  /**
   * @brief Total number of DOFs on all of the ranks.
   */
  unsigned n_global_DOFs_on_all_ranks;

  //
  //
  /**
   * @brief Number of DOFs on this rank, which are either owned or
   * not owned by this CPU.
   */
  unsigned n_local_DOFs_on_this_rank;

  //
  //
  /**
   * The @c ith member of this std::vector contains the number of
   * DOFs, which are owned by the current rank and are connected to the
   * @c ith DOF.
   */
  std::vector<int> n_local_DOFs_connected_to_DOF;

  //
  //
  /**
   * The @c ith member of this std::vector contains the number of
   * DOFs, which are NOT owned by the current rank and are connected to the
   * @c ith DOF.
   */
  std::vector<int> n_nonlocal_DOFs_connected_to_DOF;

  //
  //
  /**
   * The global number of each DOF which are present on the current
   * rank (either owned by this rank or not).
   */
  std::vector<int> scatter_from;

  //
  //
  /**
   * The local number of each DOF which are present on the current
   * rank (either owned by this rank or not).
   */
  std::vector<int> scatter_to;
};

//
//
/**
 * This class enumerate the unknowns in a model which contains
 * hybridized DG elements.
 */
template <int dim, int spacedim = dim>
struct implicit_HDG_dof_numbering : public dof_numbering<dim, spacedim>
{
  //
  //
  /**
   * @brief The constructor of the class.
   */
  implicit_HDG_dof_numbering();

  //
  //
  /**
   * @brief The destructor of the class.
   */
  ~implicit_HDG_dof_numbering();

  //
  //
  /**
   *
   */
  static model_options::options options();

  //
  //
  /**
   * This function counts the unknowns according to the model type and
   * model equation. It is called from Model::count_globals.
   */
  template <typename ModelType, typename ModelEq>
  void count_globals(ModelType *model);
};
}

#include "../../source/models/dof_numbering.cpp"

#endif
