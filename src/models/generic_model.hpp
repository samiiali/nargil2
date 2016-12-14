//#include "solution_manager.hpp"
//#include "support_classes.hpp"
#include <vector>

#ifndef GENERIC_DOF_NUM_HPP
#define GENERIC_DOF_NUM_HPP

template <int dim>
struct generic_dof_numbering
{
  generic_dof_numbering();
  virtual ~generic_dof_numbering();

  unsigned comm_rank;

  unsigned n_global_DOFs_rank_owns;
  unsigned n_global_DOFs_on_all_ranks;
  unsigned n_local_DOFs_on_this_rank;
  std::vector<int> n_local_DOFs_connected_to_DOF;
  std::vector<int> n_nonlocal_DOFs_connected_to_DOF;
  std::vector<int> scatter_from, scatter_to;

  unsigned get_global_mat_block_size();
};

#include "generic_model.cpp"

#include "hybridized_models/implicit_numbering.hpp"

#endif
