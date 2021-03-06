#include <functional>
#include <map>
#include <string>

#include <mpi.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/tria_accessor.h>

#include "../misc/utils.hpp"

#ifndef GENERIC_SOL_HPP
#define GENERIC_SOL_HPP

namespace nargil
{
/**
 *
 * This class is the wrapper around a deal.II triangulation. Since, there was
 * a chanse that I move to other libraries, I kept this wrapper as part of the
 * program. Besides the deal.II triangulation, this class also includes mpi
 * communicator and an std::map which maps deal.II cell_id to an integer.
 *
 */
template <int dim, int spacedim = dim> struct mesh
{
  /**
   *
   * Constructor for a mesh object. Obvioudly, we do not generate the mesh
   * in the constructor (although it makes a lot of sense). So, the members
   * Mesh::tria, Mesh::cell_ID_to_num, Mesh::n_ghost_cell, and
   * Mesh::n_owned_cell are unassigned at the creation time.
   * @todo Add assertions to make sure we do not use Mesh::tria, and
   * Mesh::cell_ID_to_num before initialization.
   *
   */
  mesh(const MPI_Comm &comm_,
       const unsigned n_threads_,
       const bool adaptive_on_);

  /**
   *
   * The class destructor.
   *
   */
  virtual ~mesh();

  /**
   *
   * This function generate the mesh using the function given as its argument.
   * It will also call init_Cell_ID_to_num.
   *
   */
  template <typename F> void generate_mesh(F f);

  /**
   *
   * The next function will initiate the cell_ID_to_num map. It is
   * called from Mesh::generate_mesh.
   *
   */
  void init_cell_ID_to_num();

  /**
   *
   * This function writes the mesh to a file. If it is 2D, the file is
   * an svg file and for 3D meshes, we write a gmsh file.
   *
   */
  void write_grid();

  /**
   *
   * Refines the mesh.
   *
   */
  template <typename BasisType>
  void refine_mesh(const unsigned n,
                   const BasisType &in_basis,
                   const dealii::DoFHandler<dim, spacedim> &dof_handler,
                   const LA::MPI::Vector &refine_solu);

  /**
   *
   * Refines the mesh.
   *
   */
  void refine_mesh_uniformly(const unsigned n);

  /**
   *
   * This function gives the integer ID of the corresponding input dealii cell.
   * When the cell is owned by this rank, then we should pass cell_is_owned as
   * true to this function; otherwise, this function assumes that the cell is a
   * ghost cell.
   *
   */
  int cell_id_to_num_finder(const dealiiTriCell<dim, spacedim> &in_dealii_cell,
                            const bool cell_is_owned) const;

  /**
   *
   * Given the string ID of the cell, this function returns the local
   * integer number associated with
   *
   */
  int cell_id_to_num_finder(const std::string &str_id,
                            const bool cell_is_owned) const;

  /**
   *
   * This function release the memory of owned_cell_ID_to_num
   * and ghost_cell_ID_to_num.
   *
   */
  void free_container();

  /**
   *
   * A pointer to the mpi communicator.
   *
   */
  const MPI_Comm *my_comm;

  /**
   *
   * This is a key to set h_adaptivity on or off in this mesh object.
   *
   */
  const bool adaptive_on;

  /**
   *
   * This is the number of shared memory (such as OMP, CUDA or TBB) threads.
   *
   */
  unsigned n_threads;

  /**
   *
   * @brief number of ghost cells.
   *
   */
  unsigned n_ghost_cell;

  /**
   *
   * @brief number of cells owned by the current rank.
   *
   */
  unsigned n_owned_cell;

  /**
   *
   * @brief A pointer to the deal.II triangulation.
   *
   */
  dealii::parallel::distributed::Triangulation<dim, spacedim> tria;

  /**
   *
   * @brief The refinement cycle of the triangulation.
   *
   */
  unsigned refn_cycle;

private:
  /**
   *
   * This is an std::map which maps the dealii cell ID of each
   * cell to the innerCPU number for that cell.
   *
   */
  std::map<std::string, int> owned_cell_ID_to_num;

  /**
   *
   * This is an std::map which maps the dealii cell ID of each
   * cell to the innerCPU number for that cell.
   *
   */
  std::map<std::string, int> ghost_cell_ID_to_num;
};
}

#include "../../source/mesh/mesh_handler.cpp"

#endif
