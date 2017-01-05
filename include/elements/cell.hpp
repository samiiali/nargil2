#include <assert.h>
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
// Forward declerations of cell to be used in cell_worker.
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
template <int dim, int spacedim> struct basis
{
  //
  //
  /**
   * @brief Constructor of the class
   */
  basis() {}

  //
  //
  /**
   * @brief Deconstrcutor of the class
   */
  ~basis() {}
};

//
//
/**
 * @brief Base class for all worker in elements
 */
template <int dim, int spacedim> struct cell_worker
{
  //
  //
  /**
   * @brief Constructor of the class
   */
  cell_worker(cell<dim, spacedim> *const);

  //
  //
  /**
   * @brief Deconstructor of the class
   */
  ~cell_worker() {}

  //
  //
  /**
   * @brief A pointer to the cell which contains the worker
   */
  cell<dim, spacedim> *const my_cell;
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
  cell(dealii_cell_type &inp_cell, const unsigned id_num_, base_model *model_);

  //
  //
  /**
   * We remove the copy constructor of this class to avoid unnecessary copies
   * (specially unintentional ones).
   */
  cell(const cell &inp_cell) = delete;

  //
  //
  /**
   * We need a move constructor, to be able to pass this class as function
   * arguments efficiently. Maybe, you say that this does not help efficiency
   * that much, but we are using it for semantic constraints.
   * \param inp_cell An object of the \c Cell_Class type which we steal its
   * guts.
   */
  cell(cell &&inp_cell) noexcept;

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
  create(dealii_cell_type &, const unsigned, BasisType *, base_model *);

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
   * A unique ID of each cell, which is taken from the dealii cell
   * corresponding to the current cell. This ID is unique in the
   * interCPU space.
   */
  std::string cell_id;

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
  base_model *my_model;
};
}

#include "../../source/elements/cell.cpp"

#endif // CELL_CLASS_HPP
