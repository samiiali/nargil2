#include <assert.h>
#include <functional>
#include <memory>
#include <type_traits>

#include <deal.II/grid/tria.h>

#include "../models/model_options.hpp"

#ifndef CELL_CLASS_HPP
#define CELL_CLASS_HPP

namespace nargil
{
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
 * @brief Base class for all other basis types.
 */
template <int dim, int spacedim> struct base_basis
{
  //
  //
  /**
   * @brief Constructor of the class
   */
  base_basis() {}

  //
  //
  /**
   * @brief Deconstrcutor of the class
   */
  virtual ~base_basis() {}
};

//
//
/**
 * @brief Base class for all cell_manager in elements
 */
template <int dim, int spacedim> struct cell_manager
{
  //
  //
  /**
   * @brief Constructor of the class
   */
  cell_manager(const cell<dim, spacedim> *);

  //
  //
  /**
   * @brief Deconstructor of the class
   */
  virtual ~cell_manager() {}

  //
  //
  /**
   * @brief A pointer to the cell which contains the cell_manager
   */
  const cell<dim, spacedim> *my_cell;
};

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
   * @brief assign_local_global_cell_data
   */
  void assign_local_global_cell_data(const unsigned &i_face,
                                     const unsigned &local_num_,
                                     const unsigned &global_num_,
                                     const unsigned &comm_rank_,
                                     const unsigned &half_range_);

  //
  //
  /**
   * @brief assign_local_cell_data
   */
  void assign_local_cell_data(const unsigned &i_face,
                              const unsigned &local_num_,
                              const int &comm_rank_,
                              const unsigned &half_range_);

  //
  //
  /**
   * @brief assign_ghost_cell_data
   */
  void assign_ghost_cell_data(const unsigned &i_face,
                              const int &local_num_,
                              const int &global_num_,
                              const unsigned &comm_rank_,
                              const unsigned &half_range_);

  //
  //
  /**
   * @brief dofs_ID_in_this_rank
   */
  std::vector<std::vector<int> > dofs_ID_in_this_rank;

  //
  //
  /**
   * @brief dofs_ID_in_all_ranks
   */
  std::vector<std::vector<int> > dofs_ID_in_all_ranks;

  //
  //
  /**
   * @brief Contains all of the boundary conditions of on the faces of this
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
  std::vector<boost::dynamic_bitset<> > dof_names_on_faces;

  //
  //
  /**
   * @brief Decides if the current face has a coarser neighbor.
   */
  std::vector<unsigned> half_range_flag;

  //
  //
  /**
   * @brief The CPU number of the processor which owns the current face.
   */
  std::vector<unsigned> face_owner_rank;

  int iii;
};

/**
 * @defgroup modelelements Model Elements
 * @brief Contains different elements.
 *
 * @defgroup modelbases Model Bases
 * @brief Contains bases for different models
 *
 * This group contains different model elements and the relevant
 * structures used to solve different model problems.
 */

//
//
/**
 * @brief Contains most of the required data about a generic
 * element in the mesh.
 * @ingroup modelelements
 */
template <int dim, int spacedim = dim> struct cell
{
  //
  //
  /**
   * @brief The deal.II cell iterator type.
   */
  typedef dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim> >
    dealii_cell_type;

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
  cell(dealii_cell_type &inp_cell,
       const unsigned id_num_,
       const base_model *model_);

  //
  //
  /**
   * @brief The destructor of the class.
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
  static std::unique_ptr<ModelEq>
  create(dealii_cell_type &, const unsigned, const BasisType &, base_model *);

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
   * @brief number of element faces = 2 * dim
   */
  const unsigned n_faces;

  //
  //
  /**
   * @brief id_num
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
   * @brief An iterator to the deal.II element corresponding to this Cell.
   */
  dealii_cell_type dealii_cell;

  //
  //
  /**
   * @brief A pointer to the BaseModel object which contains this Cell.
   */
  const base_model *my_model;
};
}

#include "../../source/elements/cell.cpp"

#endif // CELL_CLASS_HPP
