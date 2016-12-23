#include <functional>

#include <mpi.h>

#include <deal.II/grid/tria_accessor.h>

#include "../models/generic_model.hpp"

#ifndef GENERIC_SOL_HPP
#define GENERIC_SOL_HPP

namespace nargil
{

template <int dim, int spacedim = dim>
struct Mesh
{
  typedef dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim> >
    dealii_cell;

  Mesh(const MPI_Comm &comm_,
       const unsigned n_threads_,
       const bool adaptive_on_);

  //  void generate_mesh(
  //    const std::function<
  //      void(dealii::parallel::distributed::Triangulation<dim, spacedim> &)>
  //      &gird_gen);

  template <typename F>
  void generate_mesh(F);

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
  dealii::parallel::distributed::Triangulation<dim, spacedim> mesh;

  virtual ~Mesh();
};
}
#include "generic_solution.cpp"

#endif
