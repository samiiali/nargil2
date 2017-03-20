#include <cstdio>
#include <fstream>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include <petsc.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

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
 *
 */
template <int dim, int spacedim = dim>
struct problem_data : public nargil::diffusion<dim, spacedim>::data
{
  /**
   * @brief pi
   */
  const double pi = M_PI;

  /**
   * @brief Constructor.
   */
  problem_data() : nargil::diffusion<dim, spacedim>::data() {}

  /**
   * @brief rhs_func.
   */
  virtual double rhs_func(const dealii::Point<spacedim> &p)
  {
    return 4 * pi * pi * sin(2 * pi * p[0]);
  }

  /**
   * @brief gD_func.
   */
  virtual double gD_func(const dealii::Point<spacedim> &p)
  {
    return sin(2 * pi * p[0]);
  }

  /**
   * @brief gN_func.
   */
  virtual dealii::Tensor<1, dim> gN_func(const dealii::Point<spacedim> &p)
  {
    return dealii::Tensor<1, dim>({-2 * pi * cos(2 * pi * p[0]), 0.0});
  }

  /**
   * @brief exact_u
   */
  virtual double exact_u(const dealii::Point<spacedim> &p)
  {
    return sin(2 * pi * p[0]);
  }

  /**
   * @brief exact_q
   */
  virtual dealii::Tensor<1, dim> exact_q(const dealii::Point<spacedim> &p)
  {
    return dealii::Tensor<1, dim>({-2 * pi * cos(2 * pi * p[0]), 0.0});
  }
};

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
    std::vector<unsigned> repeats(dim, 32);
    dealii::Point<spacedim> point_1, point_2;
    point_1 = {-1.0, -1.0};
    point_2 = {1.0, 1.0};
    dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, repeats,
                                                      point_1, point_2, true);
    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::parallel::distributed::Triangulation<
        dim>::cell_iterator> >
      periodic_faces;
    dealii::GridTools::collect_periodic_faces(the_mesh, 0, 1, 0, periodic_faces,
                                              dealii::Tensor<1, dim>({2., 0.}));
    dealii::GridTools::collect_periodic_faces(the_mesh, 2, 3, 0, periodic_faces,
                                              dealii::Tensor<1, dim>({0., 2.}));
    the_mesh.add_periodicity(periodic_faces);
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
          in_manager->BCs[i_face] = nargil::boundary_condition::periodic;
          in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 1);
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

  static void run(int argc, char **argv)
  {
    PetscInitialize(&argc, &argv, NULL, NULL);
    dealii::MultithreadInfo::set_thread_limit(1);

    {
      const MPI_Comm &comm = PETSC_COMM_WORLD;
      int comm_rank, comm_size;
      MPI_Comm_rank(comm, &comm_rank);
      MPI_Comm_size(comm, &comm_size);

      nargil::mesh<2> mesh1(comm, 1, true);

      problem_data<2> data1;

      mesh1.generate_mesh(adaptive_mesh_gen);
      BasisType basis1(2, 3);
      nargil::implicit_hybridized_numbering<2> dof_counter1;
      nargil::hybridized_model_manager<2> model_manager1;

      for (unsigned i_cycle = 0; i_cycle < 1; ++i_cycle)
      {
        mesh1.init_cell_ID_to_num();
        ModelType model1(mesh1);
        model1.init_model_elements(&basis1);
        model_manager1.form_dof_handlers(&model1, &basis1);

        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::assign_BCs, assign_BCs);
        model_manager1.apply_on_ghost_cells(
          &model1, CellManagerType::assign_BCs, assign_BCs);
        dof_counter1.count_globals<BasisType>(&model1);
        //
        model_manager1.apply_on_owned_cells(&model1, ModelEq::assign_data,
                                            &data1);
        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::set_source_and_BCs);
        //
        int solver_keys = nargil::solvers::solver_props::spd_matrix;
        int update_keys = nargil::solvers::solver_update_opts::update_mat |
                          nargil::solvers::solver_update_opts::update_rhs;
        //
        nargil::solvers::petsc_direct_solver<2> solver1(solver_keys,
                                                        dof_counter1, comm);
        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::assemble_globals, &solver1);
        //
        Vec sol_vec2;
        solver1.finish_assemble(update_keys);
        solver1.form_factors();
        solver1.solve_system(&sol_vec2);
        std::vector<double> local_sol_vec(
          solver1.get_local_part_of_global_vec(&sol_vec2));
        //
        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::compute_local_unkns, local_sol_vec.data());
        //
        nargil::distributed_vector<2> dist_sol_vec(
          model_manager1.local_dof_handler, PETSC_COMM_WORLD);
        nargil::distributed_vector<2> dist_refn_vec(
          model_manager1.refn_dof_handler, PETSC_COMM_WORLD);
        //
        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::fill_viz_vector, &dist_sol_vec);

        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::fill_refn_vector, &dist_refn_vec);

        LA::MPI::Vector global_sol_vec;
        LA::MPI::Vector global_refn_vec;

        dist_sol_vec.copy_to_global_vec(global_sol_vec);
        dist_refn_vec.copy_to_global_vec(global_refn_vec);

        CellManagerType::visualize_results(model_manager1.local_dof_handler,
                                           global_sol_vec, i_cycle);

        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::interpolate_to_interior);
        std::vector<double> sum_of_L2_errors(2, 0);
        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::compute_errors, &sum_of_L2_errors);

        double u_error_global, q_error_global;
        MPI_Reduce(&sum_of_L2_errors[0], &u_error_global, 1, MPI_DOUBLE,
                   MPI_SUM, 0, comm);
        MPI_Reduce(&sum_of_L2_errors[1], &q_error_global, 1, MPI_DOUBLE,
                   MPI_SUM, 0, comm);

        if (comm_rank == 0)
        {
          std::cout << sqrt(u_error_global) << " " << sqrt(q_error_global)
                    << std::endl;
        }

        mesh1.refine_mesh(1, basis1, model_manager1.refn_dof_handler,
                          global_refn_vec);
      }
    }
    //

    PetscFinalize();
  }
};
