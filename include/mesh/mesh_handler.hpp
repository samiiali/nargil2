#include <functional>
#include <map>
#include <string>

#include <mpi.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/grid/tria_accessor.h>

#ifndef GENERIC_SOL_HPP
#define GENERIC_SOL_HPP

namespace nargil
{

template <int dim, int spacedim = dim> struct Mesh
{
  typedef dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim> >
    dealii_cell_type;

  Mesh(const MPI_Comm &comm_,
       const unsigned n_threads_,
       const bool adaptive_on_);

  template <typename F> void generate_mesh(F);

  void init_cell_ID_to_num();

  const MPI_Comm *comm;
  const bool adaptive_on;
  unsigned n_threads;

  /**
   * \brief An \c std::map which maps the dealii cell ID of each
   * cell to the innerCPU number for that cell.
   */
  std::map<std::string, int> cell_ID_to_num;

  unsigned n_ghost_cell;
  unsigned n_owned_cell;
  dealii::parallel::distributed::Triangulation<dim, spacedim> dealii_mesh;

  virtual ~Mesh();
};
}

template <int dim>
void write_grid(
  const dealii::parallel::distributed::Triangulation<dim> &the_grid,
  const MPI_Comm &comm);

#include "../../source/mesh/mesh_handler.cpp"

#endif
