//#include "solution_manager.hpp"
//#include "support_classes.hpp"
#include <vector>

#ifndef GENERIC_DOF_NUM_HPP
#define GENERIC_DOF_NUM_HPP

enum class BC
{
  not_set = ~(1 << 0),
  essential = 1 << 0,
  flux_bc = 1 << 1,
  periodic = 1 << 2,
  in_out_BC = 1 << 3,
  inflow_BC = 1 << 4,
  outflow_BC = 1 << 5,
  solid_wall = 1 << 6,
};

enum class time_integration_type
{
  implicit_type,
  explicit_type
};

enum class adaptivity_type
{
  h_adaptive,
  hp_adaptive
};

enum class cell_basis_function_type
{
  nodal_polynomial,
  modal_polynomial
};

enum class dof_numbering_type
{
  CG,
  interior_DG,
  hybridized_DG
};

template <int dim, int spacedim = dim>
struct dof_numbering
{
  dof_numbering();
  virtual ~dof_numbering();

  unsigned comm_rank;

  unsigned n_global_DOFs_rank_owns;
  unsigned n_global_DOFs_on_all_ranks;
  unsigned n_local_DOFs_on_this_rank;
  std::vector<int> n_local_DOFs_connected_to_DOF;
  std::vector<int> n_nonlocal_DOFs_connected_to_DOF;
  std::vector<int> scatter_from, scatter_to;

  unsigned get_global_mat_block_size();
};

template <int dim, int spacedim = dim>
struct implicit_hybridized_numbering : public dof_numbering<dim, spacedim>
{
  implicit_hybridized_numbering() {}
  ~implicit_hybridized_numbering() {}
};

#include "generic_model.cpp"

#include "hybridized_models/implicit_numbering.hpp"

#endif
