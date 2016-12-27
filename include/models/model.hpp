#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "../elements/cell.hpp"

#ifndef GENERIC_DOF_NUM_HPP
#define GENERIC_DOF_NUM_HPP

namespace nargil
{

//
//
// Forward decleration of Mesh
template <int dim, int spacedim> struct Mesh;

//
//
/**
 * Thid enum includes all model options.
 */
namespace ModelOptions
{
enum Options
{
  implicit_time_integration = 1 << 0,
  explicit_time_integration = 1 << 1,
  h_adaptive = 1 << 2,
  hp_adaptive = 1 << 3,
  nodal_polynomial = 1 << 4,
  modal_polynomial = 1 << 5,
  CG_dof_numbering = 1 << 6,
  LDG_dof_numbering = 1 << 7,
  HDG_dof_numbering = 1 << 8
};
}

//
//
//
// This is a forward decleration. The documentation is included with
// the main class decleration.
template <int dim, int spacedim = dim> struct implicit_HDG_dof_numbering;

//
//
/**
 * The base class for
 */
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

//
//
// The default value for spacedim parameter is given on the top.
template <int dim, int spacedim>
struct implicit_HDG_dof_numbering : public dof_numbering<dim, spacedim>
{
  implicit_HDG_dof_numbering();
  ~implicit_HDG_dof_numbering();
};

//
//
/**
 * This is an abstract model that all other models will be based on it.
 * Thus it palys the rule of an interface for the developers (and not the
 * users).
 */
struct BaseModel
{
  BaseModel();
  ~BaseModel();
};

//
//
/**
 *
 */
template <typename ModelEq, int dim, int spacedim = dim>
struct Model : public BaseModel
{
  Model(Mesh<dim, spacedim> *const);

  ~Model();

  void set_dof_numbering_type(const ModelOptions::Options);

  void init_mesh_containers();

  void free_containers();

  void set_boundary_indicator();

  void count_globals();

  Mesh<dim, spacedim> *mesh;

  std::vector<std::unique_ptr<Cell<dim, spacedim> > > all_owned_cells;

  std::unique_ptr<dof_numbering<dim, spacedim> > dof_counter;
};
}

#include "../../source/models/model.cpp"

#include "hybridized_DG.hpp"

#endif
