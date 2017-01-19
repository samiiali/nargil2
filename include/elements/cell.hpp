#include <assert.h>
#include <bitset>
#include <functional>
#include <memory>
#include <type_traits>

#include <deal.II/grid/tria.h>

#include "../models/model_options.hpp"

#ifndef CELL_CLASS_HPP
#define CELL_CLASS_HPP

namespace nargil
{
/**
 * @defgroup CellManagers Cell Managers
 * @brief All cell manager structures.
 *
 * @defgroup modelelements Model Elements
 * @brief Contains different elements.
 *
 * @defgroup modelbases Model Bases
 * @brief Contains bases for different models
 */

//
//
// Forward declerations of cell to be used in cell_manager.
template <int dim, int spacedim> struct cell;

//
//
// Forward decleration of BaseModel to be used in cell.
struct base_model;

//
//
/**
 * Base class for all other basis types.
 */
template <int dim, int spacedim> struct base_basis
{
  //
  //
  /**
   * Constructor of the class
   */
  base_basis() {}

  //
  //
  /**
   * Deconstrcutor of the class
   */
  virtual ~base_basis() {}
};

//
//
//
//
/**
 * Base class for all cell_manager in elements.
 *
 * @ingroup CellManagers
 */
template <int dim, int spacedim> struct cell_manager
{
  //
  //
  /**
   * Constructor of the class
   */
  cell_manager(const cell<dim, spacedim> *);

  //
  //
  /**
   * Deconstructor of the class
   */
  virtual ~cell_manager() {}

  //
  //
  /**
   * A pointer to the cell which contains the cell_manager
   */
  const cell<dim, spacedim> *my_cell;
};

//
//
//
//
/**
 * This is the base class for all nargil::cell_manager 's with
 * their global unknowns on their faces.
 *
 * @ingroup CellManagers
 */
template <int dim, int spacedim>
struct hybridized_cell_manager : public cell_manager<dim, spacedim>
{
  //
  //
  /**
   * The constructor of the class
   */
  hybridized_cell_manager(const cell<dim, spacedim> *);

  //
  //
  /**
   * The destructor of the class.
   */
  virtual ~hybridized_cell_manager();

  //
  //
  /**
   *
   */
  void set_cell_properties(const unsigned i_face, const unsigned in_comm_rank,
                           const unsigned in_half_range);

  //
  //
  /**
   * assign_local_global_cell_data
   */
  void set_owned_unkn_ids(const unsigned i_face,
                          const unsigned local_num_,
                          const unsigned global_num_,
                          const std::vector<unsigned> &n_unkns_per_dof);

  //
  //
  /**
   * assign_local_cell_data
   */
  void set_local_unkn_ids(const unsigned i_face,
                          const unsigned local_num_,
                          const std::vector<unsigned> &n_unkns_per_dof);

  //
  //
  /**
   * assign_ghost_cell_data
   */
  void set_ghost_unkn_ids(const unsigned i_face, const int ghost_num_,
                          const std::vector<unsigned> &n_unkns_per_dof);

  //
  //
  /**
   * assign_ghost_cell_data
   */
  void set_nonlocal_unkn_ids(const unsigned i_face, const int ghost_num_,
                             const std::vector<unsigned> &n_unkns_per_dof);

  //
  //
  /**
   * This function is called after the loop in
   * hybridized_dof_counter::count_globals() to offset the global numbers of
   * unknowns on owned and ghost cells.
   */
  void offset_global_unkn_ids(const int);

  //
  //
  /**
   * This function is used when counting the unknowns of the model. Since
   * we should number the unknown only once, if a common face of two elements
   * is visited once, we should not recount its unknowns again.
   */
  bool face_is_not_visited(const unsigned);

  //
  //
  /**
   *
   */
  unsigned get_n_open_unknowns_on_face(const unsigned,
                                       const std::vector<unsigned> &);

  //
  //
  /**
   * The local integer ID of the unknowns in this rank.
   */
  std::vector<std::vector<int> > unkns_id_in_this_rank;

  //
  //
  /**
   * The global integer ID of the unknowns in this rank.
   */
  std::vector<std::vector<int> > unkns_id_in_all_ranks;

  //
  //
  /**
   * Contains all of the boundary conditions of on the faces of this
   * Cell.
   */
  std::vector<boundary_condition> BCs;

  //
  //
  /**
   * We want to know which degrees of freedom are restrained and which are
   * open. Hence, we store a bitset which has its size equal to the number of
   * dofs of each face of the cell and it is 1 if the dof is open, and 0 if it
   * is restrained.
   */
  std::vector<boost::dynamic_bitset<> > dof_status_on_faces;

  //
  //
  /**
   * A bitset storing if the face has been counted during the unknown
   * counting procedure in for example
   * implicit_hybridized_numbering::count_globals().
   */
  std::bitset<2 * dim> face_visited;

  //
  //
  /**
   * Decides if the current face has a coarser neighbor.
   */
  std::vector<unsigned> half_range_flag;

  //
  //
  /**
   * The CPU number of the processor which owns the current face.
   */
  std::vector<unsigned> face_owner_rank;
};

//
//
//
//
/**
 * Contains most of the required data about a generic
 * element in the mesh.
 *
 * @ingroup modelelements
 */
template <int dim, int spacedim = dim> struct cell
{
  //
  //
  /**
   * The deal.II cell iterator type.
   */
  typedef dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim> >
    dealiiCell;

  //
  //
  /**
   * The type of iterator for a vector of unique_ptr's to elements.
   */
  typedef typename std::vector<std::unique_ptr<cell> >::iterator CellIterType;

  //
  //
  /**
   * We remove the default constructor to avoid uninitialized creation of Cell
   * objects.
   */
  cell() = delete;

  //
  //
  /**
   * The constructor of this class takes a deal.II cell and creates the cell.
   */
  cell(dealiiCell &inp_cell, const unsigned id_num_, const base_model *model_);

  //
  //
  /**
   * The destructor of the class.
   */
  virtual ~cell();

  //
  //
  /**
   * This is the factory function which creates a cell of type
   * ModelEq (the template parameter). This function is called by
   * Model::init_mesh_containers.
   */
  template <typename ModelEq, typename BasisType>
  static std::unique_ptr<ModelEq> create(dealiiCell &, const unsigned,
                                         const BasisType &, base_model *);

  //
  //
  /**
   * Updates the FEValues which are connected to the current element
   * (not the FEFaceValues.)
   */
  void reinit_cell_fe_vals();

  //
  //
  /**
   * Updates the FEFaceValues which are connected to a given face of
   * the current element.
   */
  void reinit_face_fe_vals(unsigned);

  //
  //
  /**
   * number of element faces = 2 * dim
   */
  const unsigned n_faces;

  //
  //
  /**
   * id_num
   */
  unsigned id_num;

  //
  //
  /**
   * A unique ID of each cell, which is taken from the dealii cell
   * corresponding to the current cell. This ID is unique in the
   * interCPU space.
   */
  std::string cell_id;

  //
  //
  /**
   * An iterator to the deal.II element corresponding to this Cell.
   */
  dealiiCell dealii_cell;

  //
  //
  /**
   * A pointer to the BaseModel object which contains this Cell.
   */
  const base_model *my_model;
};
}

#include "../../source/elements/cell.cpp"

#endif // CELL_CLASS_HPP
