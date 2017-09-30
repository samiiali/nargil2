#include <cstdio>
#include <fstream>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include <petsc.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_boundary_lib.h>

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
 *
 */
template <int dim, int spacedim = dim>
struct diffusion_data_2 : public nargil::diffusion<dim, spacedim>::data
{
  /**
   * @brief pi
   */
  const static double pi = M_PI;

  /**
   * @brief Constructor.
   */
  diffusion_data_2() : nargil::diffusion<dim, spacedim>::data() {}

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

  /**
   *
   */
  virtual dealii::Tensor<2, dim> kappa_inv(const dealii::Point<spacedim> &)
  {
    dealii::Tensor<2, dim> result;
    result[0][0] = result[1][1] = 1.;
    return dealii::invert(result);
  }

  /**
   *
   */
  virtual double tau(const dealii::Point<spacedim> &)
  {
    //
    return 0.1;
    //
  }
};

/**
 * Just a sample problem
 */
template <int dim, int spacedim = dim> struct TestProblem1
{
  typedef nargil::diffusion<dim> DiffEq;
  typedef nargil::model<DiffEq, dim> DiffModel;
  typedef typename DiffEq::hdg_polybasis DiffBasis;
  typedef typename nargil::diffusion<dim>::template hdg_manager<DiffBasis>
    DiffManagerType;

  /**
   * @brief adaptive_mesh_gen
   */
  static void rect_mesh_gen_1(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    std::vector<unsigned> repeats(dim, 64);
    dealii::Point<spacedim> point_1, point_2;
    point_1 = {-1.0, -1.0};
    point_2 = {1.0, 1.0};
    dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, repeats,
                                                      point_1, point_2, true);
    // Adding periodicity.
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
  static void assign_diff_BCs(DiffManagerType *in_manager)
  {
    unsigned n_dof_per_face = DiffBasis::get_n_dofs_per_face();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      auto &&face = in_manager->my_cell->my_dealii_cell->face(i_face);
      if (face->at_boundary())
      {
        if (fabs(face->center()[0]) > 1 - 1.E-4)
        {
          in_manager->BCs[i_face] =
            nargil::diffusion<dim>::boundary_condition::natural;
          in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 1);
        }
        else
        {
          in_manager->BCs[i_face] =
            nargil::diffusion<dim>::boundary_condition::essential;
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

  static void run(int argc, char **argv)
  {

    static_assert(dim == 2, "dim should be equal to 2.");

    PetscInitialize(&argc, &argv, NULL, NULL);
    dealii::MultithreadInfo::set_thread_limit(1);

    {
      const MPI_Comm &comm = PETSC_COMM_WORLD;
      int comm_rank, comm_size;
      MPI_Comm_rank(comm, &comm_rank);
      MPI_Comm_size(comm, &comm_size);

      nargil::mesh<dim> mesh1(comm, 1, true);

      diffusion_data_2<dim> data1;

      mesh1.generate_mesh(rect_mesh_gen_1);
      DiffBasis basis0(3, 4);
      nargil::implicit_hybridized_numbering<dim> dof_counter0;
      nargil::hybridized_model_manager<dim> model_manager0;

      for (unsigned i_cycle = 0; i_cycle < 1; ++i_cycle)
      {
        mesh1.init_cell_ID_to_num();
        DiffModel model0(mesh1);
        model0.init_model_elements(&basis0);
        model_manager0.form_dof_handlers(&model0, &basis0);

        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::assign_BCs, assign_diff_BCs);
        model_manager0.apply_on_ghost_cells(
          &model0, DiffManagerType::assign_BCs, assign_diff_BCs);
        dof_counter0.template count_globals<DiffBasis>(&model0);
        //
        model_manager0.apply_on_owned_cells(&model0, DiffEq::assign_data,
                                            &data1);
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::set_source_and_BCs);
        //
        int solver_keys0 = nargil::solvers::solver_props::default_option;
        int update_keys0 = nargil::solvers::solver_update_opts::update_mat |
                           nargil::solvers::solver_update_opts::update_rhs;
        //
        //         nargil::solvers::petsc_implicit_cg_solver<dim> solver1(
        //           solver_keys, dof_counter1, comm);
        nargil::solvers::petsc_direct_solver<dim> solver0(solver_keys0,
                                                          dof_counter0, comm);
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::assemble_globals, &solver0);

        //
        Vec sol_vec0;
        solver0.finish_assemble(update_keys0);
        solver0.form_factors();
        solver0.solve_system(&sol_vec0);
        std::vector<double> local_sol_vec0(
          solver0.get_local_part_of_global_vec(&sol_vec0));
        //
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::compute_local_unkns, local_sol_vec0.data());
        //
        nargil::distributed_vector<dim> dist_sol_vec0(
          model_manager0.local_dof_handler, PETSC_COMM_WORLD);
        nargil::distributed_vector<dim> dist_refn_vec(
          model_manager0.refn_dof_handler, PETSC_COMM_WORLD);
        //
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::fill_viz_vector, &dist_sol_vec0);

        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::fill_refn_vector, &dist_refn_vec);

        LA::MPI::Vector global_sol_vec0;
        LA::MPI::Vector global_refn_vec;

        dist_sol_vec0.copy_to_global_vec(global_sol_vec0);
        dist_refn_vec.copy_to_global_vec(global_refn_vec);

        //
        // We prepare the visulization data
        //
        std::string cycle_string = std::to_string(i_cycle);
        cycle_string =
          std::string(2 - cycle_string.length(), '0') + cycle_string;
        typename DiffEq::viz_data viz_data0(
          comm, &model_manager0.local_dof_handler, &global_sol_vec0,
          "solution-" + cycle_string, "Head", "Flow");

        // Now we visualize the results
        DiffManagerType::visualize_results(viz_data0);

        // We interpolated exact u and q to u_exact and q_exact
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::interpolate_to_interior);
        //
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::fill_viz_vec_with_exact_sol,
          &dist_sol_vec0);
        dist_sol_vec0.copy_to_global_vec(global_sol_vec0);
        typename DiffEq::viz_data viz_data2(
          comm, &model_manager0.local_dof_handler, &global_sol_vec0,
          "exact_sol-" + cycle_string, "Head", "Flow");
        DiffManagerType::visualize_results(viz_data2);

        std::vector<double> sum_of_diff_L2_errors(2, 0);
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::compute_errors, &sum_of_diff_L2_errors);

        double u_error_global, q_error_global;
        MPI_Reduce(&sum_of_diff_L2_errors[0], &u_error_global, 1, MPI_DOUBLE,
                   MPI_SUM, 0, comm);
        MPI_Reduce(&sum_of_diff_L2_errors[1], &q_error_global, 1, MPI_DOUBLE,
                   MPI_SUM, 0, comm);

        if (comm_rank == 0)
        {
          std::cout << sqrt(u_error_global) << " " << sqrt(q_error_global)
                    << std::endl;
        }

        mesh1.refine_mesh(1, basis0, model_manager0.refn_dof_handler,
                          global_refn_vec);
      }
      //
      //
    }
    //

    PetscFinalize();
  }
};
