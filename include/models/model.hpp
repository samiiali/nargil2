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
//
// Forward decleration of Mesh
template <int dim, int spacedim> struct Mesh;

//
//
//
namespace ModelOptions
{
/**
 * This enum includes all model options. Obviously, some of these options
 * cannot be used together; e.g. a model cannot have both implicit and
 * explicit time integration.
 * @todo Write functions which are asserting the consistency of the model
 * options assigned by the user.
 */
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
template <int dim, int spacedim> struct implicit_HDG_dof_numbering;

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
 * The model class contains is actually used to solve a model problem.
 */
template <int dim, int spacedim = dim> struct Model : public BaseModel
{
  //
  //
  /**
   * @brief This typename is used to count and iterate over the deal.II cells.
   */
  typedef dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim> >
    dealii_cell_type;

  //
  //
  /**
   * @brief Constructor of the class.
   */
  Model(Mesh<dim, spacedim> *const);

  //
  //
  /**
   * @brief Destructor of the class.
   */
  ~Model();

  //
  //
  /**
   * This function sets how we should choose the Model::dof_counter.
   * Obviously, we do not set the Model::dof_counter at the construction
   * stage (although it looks like a good idea).
   *
   * @todo We should assert if dof_counter is set here, before using
   * it anywhere else.
   */
  void set_dof_numbering_type(const ModelOptions::Options);

  //
  //
  /**
   * This function initiates the member Model::all_owned_cells, based on the
   * equation that we want to solve.
   */
  template <typename ModelEq> void assign_model_features();

  //
  //
  /**
   * @brief This function frees the memory used by the model.
   */
  void free_containers();

  //
  //
  /**
   * @brief Here, we set the boundary indicator of each face on the boundary.
   */
  void set_boundary_indicator();

  //
  //
  /**
   * @brief Here we count the global DOFs of the mesh.
   */
  void count_globals();

  //
  //
  /**
   * @brief This is a pointer to the mesh that the model is working on.
   */
  Mesh<dim, spacedim> *mesh;

  //
  //
  /**
   * This is a std::vector containing all of the Cell classes in the model.
   */
  std::vector<std::unique_ptr<Cell<dim, spacedim> > > all_owned_cells;

  //
  //
  /**
   * This is an instance of the dof_numbering class.
   */
  std::unique_ptr<dof_numbering<dim, spacedim> > dof_counter;
};
}

#include "../../source/models/model.cpp"

#include "hybridized_DG.hpp"

#endif
