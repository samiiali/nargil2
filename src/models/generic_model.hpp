#include "../elements/cell_class.hpp"
#include <functional>
#include <map>
#include <memory>

//#include "solution_manager.hpp"
//#include "support_classes.hpp"
#include <vector>

#ifndef GENERIC_DOF_NUM_HPP
#define GENERIC_DOF_NUM_HPP

namespace ModelOptions
{
enum time_integration_type
{
  implicit_type,
  explicit_type
};

enum adaptivity_type
{
  h_adaptive,
  hp_adaptive
};

enum cell_basis_function_type
{
  nodal_polynomial,
  modal_polynomial
};

enum dof_numbering_type
{
  CG,
  interior_DG,
  hybridized_DG
};
}

template <int dim, int spacedim = dim> struct implicit_HDG_dof_numbering;

template <int dim, int spacedim = dim> struct dof_numbering
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

// The default value for spacedim parameter is given on the top.
template <int dim, int spacedim>
struct implicit_HDG_dof_numbering : public dof_numbering<dim, spacedim>
{
  implicit_HDG_dof_numbering() {}
  ~implicit_HDG_dof_numbering() {}
};

template <int dim, int spacedim = dim> struct cell_container
{
  cell_container() {}
  ~cell_container() {}

  void set_dof_numbering(ModelOptions::time_integration_type time_integratorm_,
                         ModelOptions::dof_numbering_type numbering_type_)
  {
    if (time_integratorm_ == ModelOptions::implicit_type &&
        numbering_type_ == ModelOptions::hybridized_DG)
    {
      dof_counter =
        std::move(std::unique_ptr<implicit_HDG_dof_numbering<dim, spacedim> >(
          new implicit_HDG_dof_numbering<dim, spacedim>()));
    }
  }

  std::vector<std::unique_ptr<Cell<dim, spacedim> > > all_owned_cells;
  std::unique_ptr<dof_numbering<dim, spacedim> > dof_counter;

  void init_mesh_containers();
  void free_containers();
  void set_boundary_indicator();
  void count_globals();
};

#include "generic_model.cpp"

#include "hybridized_models/implicit_numbering.hpp"

#endif
