#include <cstdio>
#include <fstream>
#include <functional>
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
struct problem_data_2 : public nargil::diffusion<dim, spacedim>::data
{
  /**
   * @brief pi
   */
  constexpr static double epsinv = 1.0e6;

  /**
   * @brief Constructor.
   */
  problem_data_2() : nargil::diffusion<dim, spacedim>::data() {}

  /**
   * @brief rhs_func.
   */
  virtual double rhs_func(const dealii::Point<spacedim> &p)
  {
    double f_func;
    double x0 = p[0];
    double y0 = p[1];
    double psi0 =
      0.0864912785478786 - 0.3573346678775552 * pow(x0, 6) -
      0.5227047152013956 * pow(y0, 2) + 0.7614750553843495 * pow(y0, 4) -
      0.11899212585523944 * pow(y0, 6) +
      pow(x0, 4) * (-0.08759857890489645 + 3.172464834637792 * pow(y0, 2)) +
      pow(x0, 2) * (0.32364759993109177 - 2.4987434336099867 * pow(y0, 2) -
                    0.7763151405537512 * pow(y0, 4)) +
      pow(x0, 2) *
        (0.4452047152013956 + 0.22311023597857396 * pow(x0, 4) -
         4.568850332306097 * pow(y0, 2) + 1.7848818878285917 * pow(y0, 4) +
         pow(x0, 2) * (1.1422125830765242 - 2.6773228317428877 * pow(y0, 2))) *
        log(x0);
    f_func = 0.0;
    if (psi0 > 0.)
      f_func = 0.9 * pow(psi0, 1. / 3.);
    if (psi0 < 0.)
      f_func = -0.9 * pow(-psi0, 1. / 3.);
    f_func = -f_func;
    //
    return f_func;
  }

  /**
   * @brief gD_func.
   */
  virtual double gD_func(const dealii::Point<spacedim> &) { return 10.0; }

  /**
   * @brief gN_func.
   */
  virtual dealii::Tensor<1, dim> gN_func(const dealii::Point<spacedim> &)
  {
    // Based on the boundary conditions, this function should not be called.
    assert(false);
    return dealii::Tensor<1, dim>();
  }

  /**
   * @brief exact_u
   */
  virtual double exact_u(const dealii::Point<spacedim> &) { return 0.; }

  /**
   * @brief exact_q
   */
  virtual dealii::Tensor<1, dim> exact_q(const dealii::Point<spacedim> &)
  {
    return dealii::Tensor<1, dim>();
  }

  /**
   *
   */
  virtual dealii::Tensor<2, dim> kappa_inv(const dealii::Point<spacedim> &p)
  {
    double x0 = p[0];
    double y0 = p[1];
    Eigen::Matrix<double, 2, 1> B;
    B << y0 *
           (1.0454094304027911248139427451111 - 6.344929669275584 * pow(x0, 4) -
            3.045900221537398 * pow(y0, 2) + 0.7139527551314367 * pow(y0, 4) +
            pow(x0, 2) *
              (4.9974868672199735 + 3.1052605622150047 * pow(y0, 2)) +
            pow(x0, 2) * (9.137700664612193 + 5.354645663485775 * pow(x0, 2) -
                          7.139527551314367 * pow(y0, 2)) *
              log(x0)),
      x0 *
        (1.0924999150635792 - 1.9208977712867574 * pow(x0, 4) -
         9.566337199526071 * pow(y0, 2) + 0.23225160672109002 * pow(y0, 4) +
         pow(x0, 2) * (0.7918182674569383 + 10.012536506808281 * pow(y0, 2)) +
         (0.8904094304027912 + 1.3386614158714438 * pow(x0, 4) -
          9.137700664612193 * pow(y0, 2) + 3.5697637756571834 * pow(y0, 4) +
          pow(x0, 2) * (4.568850332306096492013787610749 -
                        10.70929132697155 * pow(y0, 2))) *
           log(x0));
    Eigen::Matrix<double, 2, 1> b = B / sqrt(B.squaredNorm());
    Eigen::Matrix2d kappa =
      Eigen::Matrix2d::Identity() + (epsinv - 1) * b * b.transpose();
    dealii::Tensor<2, dim> inv_of_kappa;
    inv_of_kappa[0][0] = kappa(1, 1);
    inv_of_kappa[0][1] = -kappa(0, 1);
    inv_of_kappa[1][0] = -kappa(1, 0);
    inv_of_kappa[1][1] = kappa(0, 0);
    inv_of_kappa =
      inv_of_kappa / (kappa(0, 0) * kappa(1, 1) - kappa(0, 1) * kappa(1, 0));
    return inv_of_kappa;
  }

  /**
   *
   */
  virtual double tau(const dealii::Point<spacedim> &)
  {
    //
    return 1.0e0;
    //
  }
};

/**
 * Just a sample problem
 */
template <int dim, int spacedim = dim> struct double_x_points
{
  typedef nargil::diffusion<dim> ModelEq;
  typedef nargil::model<ModelEq, dim> ModelType;
  typedef typename ModelEq::hdg_polybasis BasisType;
  typedef typename nargil::diffusion<dim>::template hdg_manager<BasisType>
    CellManagerType;

  constexpr static double epsinv = problem_data_2<dim, spacedim>::epsinv;

  /**
   * @brief u_refn_crit
   * @param i_manager
   * @return
   */
  static Eigen::VectorXd u_refn_crit(const CellManagerType *i_manager)
  {
    return i_manager->u_vec;
  }

  /**
   *
   */
  static Eigen::VectorXd q_mag_refn_crit(const CellManagerType *i_manager)
  {
    unsigned n_dofs_per_cell = i_manager->u_vec.rows();
    Eigen::VectorXd crit_vec = Eigen::VectorXd::Zero(n_dofs_per_cell);
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      for (unsigned i1 = 0; i1 < n_dofs_per_cell; ++i1)
        crit_vec[i1] += pow(i_manager->q_vec[i1 + i_dim * n_dofs_per_cell], 2);
    for (unsigned i1 = 0; i1 < n_dofs_per_cell; ++i1)
      crit_vec[i1] = sqrt(crit_vec[i1]);
    return crit_vec;
  }

  /**
   *
   */
  static Eigen::VectorXd q_dot_b_refn_crit(const CellManagerType *i_manager)
  {
    unsigned n_dofs_per_cell = i_manager->u_vec.rows();
    dealii::Tensor<1, dim> b_at_center =
      b_components(i_manager->my_cell->my_dealii_cell->center());
    Eigen::VectorXd crit_vec = Eigen::VectorXd::Zero(n_dofs_per_cell);
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      for (unsigned i1 = 0; i1 < n_dofs_per_cell; ++i1)
        crit_vec[i1] +=
          i_manager->q_vec[i1 + i_dim * n_dofs_per_cell] * b_at_center[i_dim];
    return crit_vec;
  }

  /**
   * @brief mesh generator
   */
  static void generate_mesh(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    std::vector<unsigned> repeats(dim, 1);
    repeats[0] = 1;
    dealii::Point<dim> point_1, point_2;
    point_1 = {0.4, -1.0};
    point_2 = {1.6, 1.0};
    dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, repeats,
                                                      point_1, point_2, true);
    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::parallel::distributed::Triangulation<
        dim>::cell_iterator> >
      periodic_faces;
    dealii::GridTools::collect_periodic_faces(
      the_mesh, 0, 1, 0, periodic_faces, dealii::Tensor<1, dim>({1.2, 0.}));
    dealii::GridTools::collect_periodic_faces(the_mesh, 2, 3, 0, periodic_faces,
                                              dealii::Tensor<1, dim>({0., 2.}));
    the_mesh.add_periodicity(periodic_faces);
    the_mesh.refine_global(3);
  }

  /**
   * @brief dofs_on_nodes
   */
  static void assign_mesh_BCs(CellManagerType *in_manager)
  {
    unsigned n_dof_per_face = BasisType::get_n_dofs_per_face();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      auto &&face = in_manager->my_cell->my_dealii_cell->face(i_face);
      if (face->at_boundary())
      {
        in_manager->BCs[i_face] = ModelEq::boundary_condition::essential;
        in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 0);
      }
      else
      {
        in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 1);
      }
    }
  }

  //
  //

  static dealii::Tensor<1, dim> b_components(const dealii::Point<spacedim> &p)
  {
    double x0 = p[0];
    double y0 = p[1];
    Eigen::Matrix<double, 2, 1> B;
    B << y0 *
           (1.0454094304027911248139427451111 - 6.344929669275584 * pow(x0, 4) -
            3.045900221537398 * pow(y0, 2) + 0.7139527551314367 * pow(y0, 4) +
            pow(x0, 2) *
              (4.9974868672199735 + 3.1052605622150047 * pow(y0, 2)) +
            pow(x0, 2) * (9.137700664612193 + 5.354645663485775 * pow(x0, 2) -
                          7.139527551314367 * pow(y0, 2)) *
              log(x0)),
      x0 *
        (1.0924999150635792 - 1.9208977712867574 * pow(x0, 4) -
         9.566337199526071 * pow(y0, 2) + 0.23225160672109002 * pow(y0, 4) +
         pow(x0, 2) * (0.7918182674569383 + 10.012536506808281 * pow(y0, 2)) +
         (0.8904094304027912 + 1.3386614158714438 * pow(x0, 4) -
          9.137700664612193 * pow(y0, 2) + 3.5697637756571834 * pow(y0, 4) +
          pow(x0, 2) * (4.568850332306096492013787610749 -
                        10.70929132697155 * pow(y0, 2))) *
           log(x0));
    Eigen::Matrix<double, 2, 1> b = B / sqrt(B.squaredNorm());
    dealii::Tensor<1, dim> b_tensor({b[0], b[1]});
    return b_tensor;
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

      problem_data_2<dim> data1;

      mesh1.generate_mesh(generate_mesh);
      mesh1.refine_mesh_uniformly(1);
      BasisType basis1(1, 2);
      nargil::implicit_hybridized_numbering<dim> dof_counter1;
      nargil::hybridized_model_manager<dim> model_manager1;

      for (unsigned i_cycle = 0; i_cycle < 6; ++i_cycle)
      {
        mesh1.init_cell_ID_to_num();
        ModelType model1(mesh1);
        model1.init_model_elements(&basis1);
        model_manager1.form_dof_handlers(&model1, &basis1);

        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::assign_BCs, assign_mesh_BCs);
        model_manager1.apply_on_ghost_cells(
          &model1, CellManagerType::assign_BCs, assign_mesh_BCs);
        dof_counter1.template count_globals<BasisType>(&model1);
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

        // nargil::solvers::petsc_implicit_cg_solver<dim> solver1(
        //  solver_keys, dof_counter1, comm);

        nargil::solvers::petsc_direct_solver<dim> solver1(solver_keys,
                                                          dof_counter1, comm);
        //
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

        // model_manager1.apply_on_owned_cells(
        //   &model1, CellManagerType::fill_refn_vector_with_criterion,
        //   u_refn_crit, &dist_refn_vec);

        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::fill_refn_vector_with_criterion,
          q_dot_b_refn_crit, &dist_refn_vec);

        LA::MPI::Vector global_sol_vec;
        LA::MPI::Vector global_refn_vec;

        dist_sol_vec.copy_to_global_vec(global_sol_vec);
        dist_refn_vec.copy_to_global_vec(global_refn_vec);

        //
        // We prepare the visulization data
        //
        //        std::string cycle_string = std::to_string(i_cycle);
        //        cycle_string =
        //          std::string(2 - cycle_string.length(), '0') + cycle_string;
        typename ModelEq::viz_data viz_data1(
          comm, &model_manager1.local_dof_handler, &global_sol_vec,
          "solution_q_dot_b", "Temperature", "Heat_flow");
        viz_data1.time_step = 0;
        viz_data1.cycle = i_cycle;
        // Now we visualize the results
        CellManagerType::visualize_results(viz_data1);

        //
        // Here, we visualize grad_u, which will be used to compute b.grad_u
        //
        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::fill_viz_vector_with_grad_u_dot_b,
          &dist_sol_vec, b_components);
        dist_sol_vec.copy_to_global_vec(global_sol_vec);
        typename ModelEq::viz_data viz_data3(
          comm, &model_manager1.local_dof_handler, &global_sol_vec, "Grad_T",
          "q_dot_b", "Grad_T");
        viz_data3.time_step = 0;
        viz_data3.cycle = i_cycle;
        CellManagerType::visualize_results(viz_data3);

        // We interpolated exact u and q to u_exact and q_exact
        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::interpolate_to_interior);
        //
        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::fill_viz_vec_with_exact_sol, &dist_sol_vec);
        dist_sol_vec.copy_to_global_vec(global_sol_vec);
        typename ModelEq::viz_data viz_data2(
          comm, &model_manager1.local_dof_handler, &global_sol_vec,
          "b_components", "Temperature", "b_components");
        viz_data2.time_step = 0;
        viz_data2.cycle = 0;
        CellManagerType::visualize_results(viz_data2);
        //

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
        dof_counter1.reset_components();
      }
      //
      //
    }
    //

    PetscFinalize();
  }
};
