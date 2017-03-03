#include <cstdio>
#include <fstream>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include <petsc.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/base/multithread_info.h>
#include <deal.II/lac/parallel_vector.h>

#include "include/elements/diffusion.hpp"
#include "include/mesh/mesh_handler.hpp"
#include "include/misc/utils.hpp"
#include "include/models/model.hpp"
#include "include/models/model_manager.hpp"
#include "include/solvers/solvers.hpp"

//
//
//
//
//

/**
 * Just a sample problem
 */
template <int dim, int spacedim = dim> struct Problem1
{
  typedef nargil::diffusion<2> ModelEq;
  typedef nargil::model<ModelEq, 2> ModelType;
  typedef ModelEq::hdg_polybasis BasisType;
  typedef nargil::diffusion<2>::hdg_manager<BasisType> CellManagerType;

  //
  //

  struct exact_sol_func : public dealii::Function<dim>
  {
    exact_sol_func() : dealii::Function<dim>(dim + 1, 0.0) {}
    virtual void vector_value(const dealii::Point<dim> &p,
                              dealii::Vector<double> &values) const
    {
      assert(values.size() == dim + 1);
      values(0) = sin(p[0]) + cos(p[1]);
      values(1) = -cos(p[0]);
      values(2) = sin(p[1]);
    }
  };

  /**
   * @brief dofs_on_nodes
   */
  static void assign_BCs(nargil::cell<dim, spacedim> *in_cell)
  {
    unsigned n_dof_per_face = BasisType::get_n_dofs_per_face();
    CellManagerType *i_manager =
      static_cast<ModelEq *>(in_cell)->template get_manager<CellManagerType>();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      auto &&face = in_cell->my_dealii_cell->face(i_face);
      if (face->at_boundary())
      {
        if (fabs(face->center()[0] > 1 - 1.E-4))
        {
          i_manager->BCs[i_face] = nargil::boundary_condition::essential;
          i_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 0);
        }
        else
        {
          i_manager->BCs[i_face] = nargil::boundary_condition::essential;
          i_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 0);
        }
      }
      else
      {
        i_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 1);
      }
    }
  }

  //
  //

  static double exact_uhat_func(const dealii::Point<2> &p)
  {
    return sin(p[0]) + cos(p[1]);
  }

  //
  //

  static std::vector<double> exact_local_func(const dealii::Point<2> &p)
  {
    std::vector<double> out(3);
    out[0] = sin(p[0]) + cos(p[1]);
    out[1] = -cos(p[0]);
    out[2] = sin(p[1]);
    return out;
  }

  //
  //

  static double f_func(const dealii::Point<2> &p)
  {
    return sin(p[0]) + cos(p[1]);
  }

  //
  //

  static std::vector<double> gN_func(const dealii::Point<2> &)
  {
    return std::vector<double>(2, 0.0);
  }

  //
  //

  static void run(int argc, char **argv)
  {
    PetscInitialize(&argc, &argv, NULL, NULL);
    dealii::MultithreadInfo::set_thread_limit(1);

    {
      const MPI_Comm &comm = PETSC_COMM_WORLD;
      int comm_rank;
      MPI_Comm_rank(comm, &comm_rank);

      nargil::mesh<2> mesh1(comm, 1, false);

      h_mesh1.generate_mesh(grid_gen2());
    }
    //
    PetscFinalize();
  }
};
