#include <string>
#include <vector>

#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/parallel_vector.h>

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

namespace LA
{
using namespace dealii::LinearAlgebraPETSc;
}

//
//
//
//
//

namespace nargil
{
/**
 *
 * A templated typedef.
 *
 */
template <int dim, int spacedim>
using baseTriCell =
  typename dealii::Triangulation<dim, spacedim>::active_cell_iterator;

/**
 *
 * Another templated typedef.
 *
 */
template <int dim, int spacedim>
using dealiiDoFCell =
  typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator;

/**
 *
 * This is a simple tokenizer function. I could not find a better
 * place to put it.
 *
 */
void Tokenize(const std::string &str_in,
              std::vector<std::string> &tokens,
              const std::string &delimiters);

//
//
//
//
//

/**
 *
 * @brief Wipes the object out of memory.
 *
 */
template <typename T> void reck_it_Ralph(T *obj) { T().swap(*obj); }

/**
 *
 *
 * This structure is a wrapper for dealii wrapper of parallel distributed
 * vectors. Basically, we first assemble the values to a local vector
 * (local_nodal_vec) and then copy its values to global vector
 * (global_nodal_vec), and after copression, we can use the global vector.
 *
 *
 */
template <int dim, int spacedim = dim> struct distributed_vector
{
  /**
   *
   * The constructor.
   *
   */
  distributed_vector(const dealii::DoFHandler<dim, spacedim> &dof_handler,
                     const MPI_Comm &comm);

  /**
   *
   * This function assembles the val in idx position of the distributed vector.
   *
   */
  void assemble(const unsigned &idx, const double &val);

  /**
   *
   * This funciton reintializes the global_vector.
   *
   */
  void reinit_global_vec(LA::MPI::Vector &global_nodal_vec);

  /**
   *
   * This function copies the values of the local vector to the global
   * vector after all the values are assembeled to the local vector
   * and compresses global vector. By default this function also
   * reinitializes the global vector (calls reinit_global_vec()).
   *
   */
  void copy_to_global_vec(LA::MPI::Vector &global_nodal_vec,
                          const bool &do_reinit_global_vec = true);

  /**
   *
   * @brief my_dof_handler
   *
   */
  const dealii::DoFHandler<dim> *const my_dof_handler;

  /**
   *
   * @brief idx_set
   *
   */
  dealii::IndexSet idx_set;

  /**
   *
   * @brief idx_vec
   *
   */
  std::vector<unsigned> idx_vec;

  /**
   *
   * @brief my_comm
   *
   */
  const MPI_Comm *const my_comm;

  /**
   *
   *  @brief local_nodal_vec
   *
   */
  LA::MPI::Vector local_nodal_vec;
};

/**
 *
 *
 * This is the derived classes to give access to the dealii cells corresponding
 * to each element. The main point of this class is to have generic_neighbor or
 * generic_... function which works for either interior cells or the cells on
 * the boundary.
 *
 *
 */
template <int dim, int spacedim = dim>
struct dealiiTriCell : public baseTriCell<dim, spacedim>
{
  /**
   *
   * @brief The constructor.
   *
   */
  dealiiTriCell();

  /**
   *
   * For the cells inside the domain this function returns the
   * regular neighbor, and for the cells on the periodic boundary it
   * returns the periodic_neighbor.
   *
   */
  dealiiTriCell generic_neighbor(const unsigned i_face);

  /**
   *
   * For the cells inside the domain this function returns if the
   * neighbor is coarser, and for the cells on the periodic boundary it
   * returns the periodic_neighbor is coarser.
   *
   */
  bool generic_neighbor_is_coarser(const unsigned i_face);

  /**
   *
   * For the cells inside the domain this function returns the
   * face number of the regular element attached to this element. For
   * the cells on the periodic boundary it returns the face number of the
   * periodic_neighbor which is attached to this cell.
   *
   */
  unsigned generic_neighbor_face_no(const unsigned i_face);

  /**
   *
   * Similar to the above two fucntions (Yes, I am lazy).
   *
   */
  dealiiTriCell generic_neighbor_child_on_subface(const unsigned i_face,
                                                  const unsigned i_subface);
}; // dealiiTriCell
}

#include "../../source/misc/utils.cpp"

#endif
