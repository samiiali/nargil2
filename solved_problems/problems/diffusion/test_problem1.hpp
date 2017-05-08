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
 * This problem checks the periodic BC on a distributed grid.
 *
 */
template <int dim, int spacedim = dim>
struct problem_data_0 : public nargil::diffusion<dim, spacedim>::data
{
  /**
   * @brief Constructor.
   */
  problem_data_0() : nargil::diffusion<dim, spacedim>::data() {}

  /**
   * @brief rhs_func.
   */
  virtual double rhs_func(const dealii::Point<spacedim> &p)
  {
    double r = p[0];
    //
    // ***
    //
    return 0.0;
    //
    // ***
    //
    // return (56.0 * pow(r, 6.0) * pow((1 - r), 16.0) +
    //         256.0 * pow(r, 7.0) * pow((1 - r), 15.0) +
    //         240.0 * pow(r, 8.0) * pow((1 - r), 14.0));
  }

  /**
   * @brief gD_func.
   */
  virtual double gD_func(const dealii::Point<spacedim> &p)
  {
    double temperature = 4.7e-6;
    if (p[0] > r_o - 1.e-6)
      temperature = 4.6e-6;
    return temperature;
  }

  /**
   * @brief gN_func.
   */
  virtual dealii::Tensor<1, dim> gN_func(const dealii::Point<spacedim> &p)
  {
    return dealii::Tensor<1, dim>({0, 0, 0});
  }

  /**
   * @brief exact_u
   */
  virtual double exact_u(const dealii::Point<spacedim> &p) { return 0.; }

  /**
   * @brief exact_q
   */
  virtual dealii::Tensor<1, dim> exact_q(const dealii::Point<spacedim> &p)
  {
    double R = 5.;
    double r_m = (r_i + r_o) / 2.;

    double y1 = p[0];
    //
    // ***
    //
    // double y2 = p[1] / p[0];
    double y2 = p[1];
    //
    double y3 = p[2] / R;

    dealii::Tensor<1, dim> B1(
      {-(pow(-1 + y1, 2) * y1 *
         (4 * sin(4 * y2 - 3 * y3) + 3 * sin(3 * y2 - 2 * y3))) /
         200.,
       ((3 - 5 * y1) * y1) / (3. * exp((10 * y1) / 3.)) -
         ((-1 + y1) * y1 * (-1 + 2 * y1) *
          (cos(4 * y2 - 3 * y3) + cos(3 * y2 - 2 * y3))) /
           100.,
       1.});

    dealii::Tensor<1, dim> b1 = B1 / sqrt(B1 * B1);
    return b1;
  }

  /**
   *
   */
  virtual dealii::Tensor<2, dim> kappa_inv(const dealii::Point<spacedim> &p)
  {

    //
    // Here we change the coordinates and use a rectangle with periodic BC
    // instead of the annulus that we have:
    //
    //
    //                              | y = r
    //                              |
    //                              |
    //                              |
    //                              |
    // r = ro -|------------------------------------------|
    //         |                    |                     |
    //         |                    |                     |
    //         |                    ----------------------|----- x
    //         |                                          |
    //         |                                          |
    // r = ri -|------------------------------------------|
    //       -pi*cl                                         pi*cl
    // cl =(ro+ri)/2
    //
    //
    // As a result r will be equivalent to y, i.e. y = r - rm, with
    // rm = (0.63 + 0.54) / 2 . Meanwhile, x is equivalent to theta
    // and x \in [-pi,pi]. That is why, we add the new function
    // generate_rect_mesh() and replace mesh_gen() in this example.
    //

    //     double r = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    //     double z = p[2];
    //     double theta = std::atan2(p[1], p[0]);

    double R = 5.;
    double r_m = (r_i + r_o) / 2.;

    double y1 = p[0];
    //
    // ***
    //
    double y2 = p[1];
    // double y2 = p[1] / p[0];
    //
    // ***
    //
    double y3 = p[2] / R;
    // double y3 = p[2];
    //

    dealii::Tensor<1, dim> B1(
      {-(pow(-1 + y1, 2) * y1 *
         (3 * sin(3 * y2 + 2 * y3) + 4 * sin(4 * y2 + 3 * y3))) /
         5000.,
       ((3 - 5 * y1) * y1) / (3. * exp((10 * y1) / 3.)) -
         ((-1 + y1) * y1 * (-1 + 2 * y1) *
          (cos(3 * y2 + 2 * y3) + cos(4 * y2 + 3 * y3))) /
           2500.,
       1.});

    dealii::Tensor<1, dim> b1 = B1 / sqrt(B1 * B1);

    dealii::Tensor<2, dim> result((epsinv - 1.) *
                                  dealii::outer_product(b1, b1));
    result[0][0] += 1.;
    //
    // ***
    //
    result[1][1] += 1. / y1 / y1;
    // result[1][1] += 1.;
    //
    result[2][2] += 1.;

    return dealii::invert(result);
  }

  /**
   *
   */
  virtual double tau(const dealii::Point<spacedim> &)
  {
    //
    return 1.0e3;
    //
  }
};

/**
 * Just a sample problem
 */
template <int dim, int spacedim = dim> struct TestProblem1
{
  typedef nargil::diffusion<dim> ModelEq;
  typedef nargil::model<ModelEq, dim> ModelType;
  typedef typename ModelEq::hdg_polybasis BasisType;
  typedef typename nargil::diffusion<dim>::template hdg_manager<BasisType>
    CellManagerType;

  const static double epsinv = problem_data_0<dim, spacedim>::epsinv;
  const static double r_i = problem_data_0<dim, spacedim>::r_i;
  const static double r_o = problem_data_0<dim, spacedim>::r_o;

  /**
   * @brief adaptive_mesh_gen
   */
  /*
  static void mesh_gen(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    dealii::CylindricalManifold<dim> manifold1(2);
    dealii::GridGenerator::cylinder_shell(the_mesh, 2. * M_PI * 5.0, r_i, r_o,
                                          15, 1);

    // Here we assign boundary id 10 and 11 to the bottom and top caps of
    // the cylindrical shell.
    for (auto &&i_cell : the_mesh.active_cell_iterators())
    {
      for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
      {
        if (i_cell->face(i_face)->at_boundary())
        {
          dealii::Point<dim> face_center = i_cell->face(i_face)->center();
          if (face_center[2] < 1.e-6)
            i_cell->face(i_face)->set_boundary_id(10);
          if (face_center[2] > 2. * M_PI * 5.0 - 1.e-6)
            i_cell->face(i_face)->set_boundary_id(11);
        }
      }
    }
    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::parallel::distributed::Triangulation<
        dim>::cell_iterator> >
      periodic_faces;
    dealii::GridTools::collect_periodic_faces(
      the_mesh, 10, 11, 2, periodic_faces,
      dealii::Tensor<1, dim>({0., 0., 2.0 * M_PI * 5.0}));
    the_mesh.add_periodicity(periodic_faces);

    the_mesh.set_all_manifold_ids(0);
    the_mesh.set_manifold(0, manifold1);
    the_mesh.refine_global(5);
    the_mesh.set_manifold(0);
  }
  */

  /**
   * @brief adaptive_mesh_gen
   */
  static void generate_rect_mesh(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    std::vector<unsigned> refine_repeats = {80, 80, 4};
    //
    // ***
    //
    // double r_m = (r_i + r_o) / 2.;
    double r_m = 1;
    //
    dealii::Point<dim> corner_1(r_i, 0., 0.);
    // dealii::Point<dim> corner_2(r_o, 2. * 0.59 * M_PI, 5. * 2. * M_PI);
    dealii::Point<dim> corner_2(r_o, 2. * M_PI * r_m, 2. * 5. * M_PI);
    dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, refine_repeats,
                                                      corner_1, corner_2, true);
    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::parallel::distributed::Triangulation<
        dim>::cell_iterator> >
      periodic_faces;
    dealii::GridTools::collect_periodic_faces(
      the_mesh, 2, 3, 1, periodic_faces,
      dealii::Tensor<1, dim>({0.0, 2.0 * M_PI * r_m, 0.}));
    dealii::GridTools::collect_periodic_faces(
      the_mesh, 4, 5, 2, periodic_faces,
      dealii::Tensor<1, dim>({0., 0., 2.0 * M_PI * 5.0}));
    the_mesh.add_periodicity(periodic_faces);
    // the_mesh.refine_global(4);
  }

  /**
   * @brief dofs_on_nodes
   */
  /*
  static void assign_BCs(CellManagerType *in_manager)
  {
    unsigned n_dof_per_face = BasisType::get_n_dofs_per_face();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      auto &&face = in_manager->my_cell->my_dealii_cell->face(i_face);
      if (face->at_boundary())
      {
        if (face->center()[2] > 2. * M_PI * 5.0 - 1.e-6 ||
            face->center()[2] < 1.e-6)
        {
          in_manager->BCs[i_face] = nargil::boundary_condition::periodic;
          in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 1);
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
  */

  /**
   * @brief dofs_on_nodes
   */
  static void assign_rect_mesh_BCs(CellManagerType *in_manager)
  {
    //
    // ***
    //
    double r_m = 1.;
    // double r_m = (r_i + r_o) / 2.;
    //
    unsigned n_dof_per_face = BasisType::get_n_dofs_per_face();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      auto &&face = in_manager->my_cell->my_dealii_cell->face(i_face);
      if (face->at_boundary())
      {
        if (face->center()[2] > 2. * 5.0 * M_PI - 1.e-6 ||
            face->center()[2] < 1.e-6 || face->center()[1] < 1.e-6 ||
            face->center()[1] > r_m * 2. * M_PI - 1.e-6)
        // face->center()[1] > 2. * M_PI - 1.e-6)
        {
          in_manager->BCs[i_face] = nargil::boundary_condition::periodic;
          in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 1);
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

  static void run(int argc, char **argv)
  {

    static_assert(dim == 3, "dim should be equal to 3.");

    PetscInitialize(&argc, &argv, NULL, NULL);
    dealii::MultithreadInfo::set_thread_limit(1);

    {
      const MPI_Comm &comm = PETSC_COMM_WORLD;
      int comm_rank, comm_size;
      MPI_Comm_rank(comm, &comm_rank);
      MPI_Comm_size(comm, &comm_size);

      nargil::mesh<dim> mesh1(comm, 1, true);

      problem_data_0<dim> data1;

      mesh1.generate_mesh(generate_rect_mesh);
      BasisType basis1(1, 2);
      nargil::implicit_hybridized_numbering<dim> dof_counter1;
      nargil::hybridized_model_manager<dim> model_manager1;

      for (unsigned i_cycle = 0; i_cycle < 1; ++i_cycle)
      {
        mesh1.init_cell_ID_to_num();
        ModelType model1(mesh1);
        model1.init_model_elements(&basis1);
        model_manager1.form_dof_handlers(&model1, &basis1);

        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::assign_BCs, assign_rect_mesh_BCs);
        model_manager1.apply_on_ghost_cells(
          &model1, CellManagerType::assign_BCs, assign_rect_mesh_BCs);
        dof_counter1.template count_globals<BasisType>(&model1);
        //
        model_manager1.apply_on_owned_cells(&model1, ModelEq::assign_data,
                                            &data1);
        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::set_source_and_BCs);
        //
        int solver_keys = nargil::solvers::solver_props::default_option;
        int update_keys = nargil::solvers::solver_update_opts::update_mat |
                          nargil::solvers::solver_update_opts::update_rhs;
        //
        //         nargil::solvers::petsc_implicit_cg_solver<dim> solver1(
        //           solver_keys, dof_counter1, comm);
        nargil::solvers::petsc_direct_solver<dim> solver1(solver_keys,
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
        nargil::distributed_vector<dim> dist_sol_vec(
          model_manager1.local_dof_handler, PETSC_COMM_WORLD);
        nargil::distributed_vector<dim> dist_refn_vec(
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

        //
        // We prepare the visulization data
        //
        std::string cycle_string = std::to_string(i_cycle);
        cycle_string =
          std::string(2 - cycle_string.length(), '0') + cycle_string;
        typename ModelEq::viz_data viz_data1(
          comm, &model_manager1.local_dof_handler, &global_sol_vec,
          "solution" + cycle_string, "Temperature", "Heat_flow");

        // Now we visualize the results
        CellManagerType::visualize_results(viz_data1);

        // We interpolated exact u and q to u_exact and q_exact
        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::interpolate_to_interior);
        //
        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::fill_viz_vec_with_exact_sol, &dist_sol_vec);
        dist_sol_vec.copy_to_global_vec(global_sol_vec);
        typename ModelEq::viz_data viz_data2(
          comm, &model_manager1.local_dof_handler, &global_sol_vec,
          "kappa_comps" + cycle_string, "Temperature", "b_components");
        CellManagerType::visualize_results(viz_data2);

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
      //
      //
    }
    //

    PetscFinalize();
  }
};
