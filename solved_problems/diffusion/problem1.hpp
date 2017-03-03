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

  /**
   * @brief adaptive_mesh_gen
   */
  static void adaptive_mesh_gen(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    std::vector<unsigned> repeats(dim, 8);
    dealii::Point<spacedim> point_1, point_2;
    point_1 = {-1.0, -1.0};
    point_2 = {1.0, 1.0};
    dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, repeats,
                                                      point_1, point_2, true);
    /*
    typedef typename nargil::mesh<dim, spacedim>::dealiiTriCell dealiiTriCell;
    dealii::Point<dim> refn_point1(-0.625, 0.125);
    dealii::Point<dim> refn_point2(0.625, -0.125);

    for (unsigned i_refn = 0; i_refn < 2; ++i_refn)
    {
      for (dealiiTriCell &&i_cell : the_mesh.active_cell_iterators())
      {
        if (i_cell->is_locally_owned() && (i_cell->point_inside(refn_point1) ||
                                           i_cell->point_inside(refn_point2)))
        {
          i_cell->set_refine_flag();
        }
      }
      the_mesh.execute_coarsening_and_refinement();
    }
    */
  }

  /**
   * @brief dofs_on_nodes
   */
  static void assign_BCs(CellManagerType *in_manager)
  {
    unsigned n_dof_per_face = BasisType::get_n_dofs_per_face();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      auto &&face = in_manager->my_cell->my_dealii_cell->face(i_face);
      if (face->at_boundary())
      {
        if (fabs(face->center()[0] > 1 - 1.E-4))
        {
          in_manager->BCs[i_face] = nargil::boundary_condition::essential;
          in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 0);
        }
        else
        {
          in_manager->BCs[i_face] = nargil::boundary_condition::essential;
          in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 0);
        }
      }
      else
      {
        in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 1);
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

      nargil::mesh<2> mesh1(comm, 1, true);

      mesh1.generate_mesh(adaptive_mesh_gen);
      BasisType basis1(2, 3);
      nargil::implicit_hybridized_numbering<2> dof_counter1;
      nargil::hybridized_model_manager<2> model_manager1;

      for (unsigned i_cycle = 0; i_cycle < 3; ++i_cycle)
      {
        mesh1.init_cell_ID_to_num();
        ModelType model1(mesh1);
        model1.init_model_elements(&basis1);
        model_manager1.form_dof_handlers(&model1, &basis1);

        model_manager1.invoke(&model1, CellManagerType::assign_BCs, assign_BCs);
        dof_counter1.count_globals<BasisType>(&model1);
        //
        model_manager1.invoke(&model1, CellManagerType::set_source_and_BCs,
                              f_func, exact_uhat_func, gN_func);
        //
        nargil::solvers::simple_implicit_solver<2> solver1(dof_counter1);
        model_manager1.invoke(&model1, CellManagerType::assemble_globals,
                              &solver1);
        //
        Eigen::VectorXd sol_vec;
        solver1.finish_assemble();
        solver1.form_factors();
        solver1.solve_system(sol_vec);
        //
        model_manager1.invoke(&model1, CellManagerType::compute_local_unkns,
                              sol_vec.data());
        //
        nargil::distributed_vector<2> dist_sol_vec(
          model_manager1.local_dof_handler, PETSC_COMM_WORLD);
        nargil::distributed_vector<2> dist_refn_vec(
          model_manager1.refn_dof_handler, PETSC_COMM_WORLD);
        //
        model_manager1.invoke(&model1, CellManagerType::fill_viz_vector,
                              &dist_sol_vec);

        model_manager1.invoke(&model1, CellManagerType::fill_refn_vector,
                              &dist_refn_vec);

        LA::MPI::Vector global_sol_vec;
        LA::MPI::Vector global_refn_vec;

        dist_sol_vec.copy_to_global_vec(global_sol_vec);
        dist_refn_vec.copy_to_global_vec(global_refn_vec);

        CellManagerType::visualize_results(model_manager1.local_dof_handler,
                                           global_sol_vec, i_cycle);

        model_manager1.invoke(&model1, CellManagerType::interpolate_to_interior,
                              exact_local_func);
        std::vector<double> sum_of_L2_errors(2, 0);
        model_manager1.invoke(&model1, CellManagerType::compute_errors,
                              &sum_of_L2_errors);
        std::cout << sqrt(sum_of_L2_errors[0]) << " "
                  << sqrt(sum_of_L2_errors[1]) << std::endl;

        mesh1.refine_mesh(1, basis1, model_manager1.refn_dof_handler,
                          global_refn_vec);
      }
    }
    //

    PetscFinalize();
  }
};
