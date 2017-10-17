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
struct problem_data_2 : public nargil::diffusion<dim, spacedim>::data
{
  /**
   * @brief pi
   */
  constexpr static double epsinv = 1.0e10;
  constexpr static double z_0 = 0.0;
  constexpr static double z_h = 2.0 * M_PI;
  constexpr static double s_i = 0.1;
  constexpr static double s_m = 1.00;
  constexpr static double s_o = 2.0 * M_PI + .1;
  /**
   * @brief Constructor.
   */
  problem_data_2() : nargil::diffusion<dim, spacedim>::data() {}

  /**
   * @brief rhs_func.
   */
  virtual double rhs_func(const dealii::Point<spacedim> &p)
  {
    //
    // ***
    //
    // return 0.0;
    //
    // ***
    //
    // double r = sqrt(p[0] * p[0] + p[1] * p[1]);
    // return (56.0 * pow(r, 6.0) * pow((1 - r), 16.0) +
    //         256.0 * pow(r, 7.0) * pow((1 - r), 15.0) +
    //         240.0 * pow(r, 8.0) * pow((1 - r), 14.0));

    double s = p[0];
    double left = 0.55;
    double dr = 0.08 / (2.0 * M_PI);

    return -56.0 * pow(dr, 2) * pow((left + dr * s), 6) *
             pow((pow((left + dr * s), 2) - 1.0), 8) -
           272.0 * pow(dr, 2) * pow((left + dr * s), 8) *
             pow((pow((left + dr * s), 2) - 1.0), 7) -
           224.0 * pow(dr, 2) * pow((left + dr * s), 10) *
             pow((pow((left + dr * s), 2) - 1.0), 6);

    // return -9.3198864878881439535914505906675e-391*pow( (9174657822756255.0*s
    // + 396316767208603648.0) ,6)
    //   *pow( (1019406424750695.0*s - 36028797018963968.0) ,6)*pow(
    //   (9174657822756255.0*s + 1116892707587883008.0), 6)
    //   *(4.8888711810514237134313323899743e65*pow(s,4) +
    //   8.4473629827059037211098258702876e67*pow(s,3)
    // 	+ 3.3756189548712634831839246221647e69*pow(s,2) -
    // 2.3618327835261228636907987719231e70*s -
    // 3.2515549176917038198854303552233e71);

    // return -(23568816926105512708339039655007*pow(
    // (pow(((1834931564551251*s)/144115188075855872 + 11/20),2) - 1),8)
    // 	     *pow( ((1834931564551251*s)/144115188075855872 + 11/20),6)
    // )/2596148429267413814265248164610048
    //   - (3366973846586501815477005665001*pow((pow(
    //   ((1834931564551251*s)/144115188075855872 + 11/20),2) - 1),7)
    // 	 *pow( ((1834931564551251*s)/144115188075855872 + 11/20),8)
    // )/1298074214633706907132624082305024
    //   -
    //   (1834931564551251*((3366973846586501815477005665001*s)/10384593717069655257060992658440192
    // 			   + 20184247210063761/1441151880758558720)*pow((pow(
    // ((1834931564551251*s)/144115188075855872 + 11/20),2) - 1),7)
    // 	 *pow( ((1834931564551251*s)/144115188075855872 +
    // 11/20),7))/1125899906842624
    //   - 56*pow(
    //   ((3366973846586501815477005665001*s)/10384593717069655257060992658440192
    // 		 + 20184247210063761/1441151880758558720),2)*pow((pow(
    // ((1834931564551251*s)/144115188075855872 + 11/20),2)
    // 								  - 1),6)*pow(
    // ((1834931564551251*s)/144115188075855872
    // +
    // 11/20),8);

    // -
    // (23568816926105512708339039655007*(((1834931564551251*s)/144115188075855872
    // + 11/20)^2 - 1)^8
    // 	 *((1834931564551251*s)/144115188075855872 +
    // 11/20)^6)/2596148429267413814265248164610048
    // -
    // (3366973846586501815477005665001*(((1834931564551251*s)/144115188075855872
    // + 11/20)^2 - 1)^7
    // 	 *((1834931564551251*s)/144115188075855872 +
    // 11/20)^8)/1298074214633706907132624082305024
    // -
    // (1834931564551251*((3366973846586501815477005665001*s)/10384593717069655257060992658440192
    // 			   +
    // 20184247210063761/1441151880758558720)*(((1834931564551251*s)/144115188075855872
    // + 11/20)^2 - 1)^7
    // 	 *((1834931564551251*s)/144115188075855872 + 11/20)^7)/1125899906842624
    // -
    // 56*((3366973846586501815477005665001*s)/10384593717069655257060992658440192
    // +
    // 20184247210063761/1441151880758558720)^2*(((1834931564551251*s)/144115188075855872
    // + 11/20)^2 - 1)^6*((1834931564551251*s)/144115188075855872 + 11/20)^8
  }

  /**
   * @brief gD_func.
   */
  virtual double gD_func(const dealii::Point<spacedim> &p)
  {

    double s = p[0], y = p[1], z = p[2];
    // double g = M_PI;
    double temperature =
      0.0; // 1.0; //1e-4*exp( -(-y-g)*(y-g) - (z-g)*(z-g) )/0.2;

    if (!(s > s_o - 1.e-6))
      assert(false);

    return temperature;
  }

  /**
   * @brief gN_func.
   */
  virtual dealii::Tensor<1, dim> gN_func(const dealii::Point<spacedim> &p)
  {
    double s = p[0];
    if (!(s < s_i + 1.E-6))
      assert(false);
    return dealii::Tensor<1, dim>({0, 0, 0});
  }

  /**
   * @brief exact_u
   */
  virtual double exact_u(const dealii::Point<spacedim> &) { return 0.; }

  /**
   * @brief exact_q
   */
  virtual dealii::Tensor<1, dim> exact_q(const dealii::Point<spacedim> &p)
  {
    double R = 5.;
    // double r_m = (r_i + r_o) / 2.;

    double y1 = sqrt(p[0] * p[0] + p[1] * p[1]);
    //
    // double y2 = p[1];
    double y2 = atan2(p[1], p[0]);
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
    //
    //           [ cos(x,r)  cos(x,t)  cos(x,z) ]
    // beta_ij = [ cos(y,r)  cos(y,t)  cos(y,z) ]
    //           [ cos(z,r)  cos(z,t)  cos(z,z) ]
    //
    dealii::Tensor<2, dim> beta_ik;
    beta_ik[0][0] = cos(y2);
    beta_ik[0][1] = -sin(y2);
    beta_ik[1][0] = sin(y2);
    beta_ik[1][1] = cos(y2);
    beta_ik[2][2] = 1.;
    // b1 = beta_ik * b1;

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

    double s = p[0], y = p[1], z = p[2];
    double R = 5.0, a = 1.0, B0 = 1.0;

    // .63-.55 = .08
    double dx = 0.08 / (2. * M_PI); // Ali, this is just x_step in equation 7
    double x = 0.55 + dx * s;

    double psitilde = 0.0002 * 4.0;
    double psishape = a * B0 * (x * x) * (1. - x) * (1. - x);
    double psishapep = 2.0 * a * B0 * x * (1. - x) * (1. - x) -
                       2.0 * a * B0 * x * x * (1. - x); //
    double psi32 = cos(3.0 * y - 2.0 * z);
    double psi43 = 0.0; // cos(4.0 * y - 3.0 * z);
    double psip32 = -3.0 * sin(3.0 * y - 2.0 * z);
    double psip43 = 0.0; //-4.0 * sin(4.0 * y - 3.0 * z);
    double psipert = psi32 + psi43;
    double psipertp = psip32 + psip43;
    double qsafety = 0.2 * exp(x / (a * 0.3));

    double bs = (psitilde * psishape * psipertp) / (B0 * x * dx);
    double by =
      (-(psitilde * psishapep * psipert / (B0 * x))) + a / (R * qsafety);

    //    double bs = bs;
    //    double by = thetaprime;
    double bz = 1.0 / R;

    dealii::Tensor<2, dim> result;

    result[0][0] = 1.0 + (epsinv - 1.) * bs * bs;
    result[0][1] = (epsinv - 1.) * bs * by;
    result[0][2] = (epsinv - 1.) * bs * bz;
    result[1][0] = result[0][1];
    result[1][1] = 1.0 / (x * x) + (epsinv - 1.) * by * by;
    // result[1][1] = 1.0 + (epsinv - 1.) * by * by;
    result[1][2] = (epsinv - 1) * by * bz;
    result[2][0] = result[0][2];
    result[2][1] = result[1][2];
    result[2][2] = 1. + (epsinv - 1.) * bz * bz;

    return dealii::invert(result);
  }

  /**
   *
   */
  virtual double tau(const dealii::Point<spacedim> &)
  {
    //
    return 1.0e1;
    //
  }
};

/**
 * Just a sample problem
 */
template <int dim, int spacedim = dim> struct Problem2
{
  typedef nargil::diffusion<dim> ModelEq;
  typedef nargil::model<ModelEq, dim> ModelType;
  typedef typename ModelEq::hdg_polybasis BasisType;
  typedef typename nargil::diffusion<dim>::template hdg_manager<BasisType>
    CellManagerType;

  constexpr static double epsinv = problem_data_2<dim, spacedim>::epsinv;
  constexpr static double s_i = problem_data_2<dim, spacedim>::s_i;
  constexpr static double s_o = problem_data_2<dim, spacedim>::s_o;
  constexpr static double z_0 = problem_data_2<dim, spacedim>::z_0;
  constexpr static double z_h = problem_data_2<dim, spacedim>::z_h;
  constexpr static double s_m = problem_data_2<dim, spacedim>::s_m;

  /**
   * @brief adaptive_mesh_gen
   */
  static void mesh_gen(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    dealii::CylindricalManifold<dim> manifold1(2);
    dealii::GridGenerator::cylinder_shell(the_mesh, 2. * M_PI * 5.0, s_i, s_o,
                                          3, 1);

    // Here we assign boundary id 10 and 11 to the bottom and top caps of
    // the cylindrical shell.
    for (auto &&i_cell : the_mesh.active_cell_iterators())
    {
      for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
      {
        if (i_cell->face(i_face)->at_boundary())
        {
          dealii::Point<dim> face_center = i_cell->face(i_face)->center();
          if (face_center[2] < 1.e-4)
            i_cell->face(i_face)->set_boundary_id(10);
          if (face_center[2] > 2. * M_PI * 5.0 - 1.e-4)
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
    the_mesh.refine_global(6);
    the_mesh.set_manifold(0);
  }

  /**
   * @brief adaptive_mesh_gen
   */
  static void generate_rect_mesh(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    std::vector<unsigned> refine_repeats = {20, 20, 20};
    //
    // ***
    //
    // double r_m = (r_i + r_o) / 2.;
    // double r_m = 1;
    //
    dealii::Point<dim> corner_1(s_i, 0., z_0);
    dealii::Point<dim> corner_2(s_o, 2. * M_PI * s_m, z_h);
    dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, refine_repeats,
                                                      corner_1, corner_2, true);
    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::parallel::distributed::Triangulation<
        dim>::cell_iterator> >
      periodic_faces;
    dealii::GridTools::collect_periodic_faces(
      the_mesh, 2, 3, 1, periodic_faces,
      dealii::Tensor<1, dim>({0.0, 2.0 * M_PI * s_m, 0.}));
    dealii::GridTools::collect_periodic_faces(
      the_mesh, 4, 5, 2, periodic_faces,
      dealii::Tensor<1, dim>({0., 0., z_h - z_0}));
    the_mesh.add_periodicity(periodic_faces);
    //    the_mesh.refine_global(2);
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
        if (face->center()[2] > z_h - 1.e-6 || face->center()[2] < 1.e-6)
        {
          in_manager->BCs[i_face] = ModelEq::boundary_condition::periodic;
          in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 1);
        }
        else
        {
          in_manager->BCs[i_face] = ModelEq::boundary_condition::essential;
          in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 0);
        }
      }
      else
      {
        in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 1);
      }
    }
  }

  /**
   * @brief dofs_on_nodes
   */
  static void assign_rect_mesh_BCs(CellManagerType *in_manager)
  {
    //
    // ***
    //
    // double r_m = 1.;
    // double r_m = (r_i + r_o) / 2.;
    //
    unsigned n_dof_per_face = BasisType::get_n_dofs_per_face();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      auto &&face = in_manager->my_cell->my_dealii_cell->face(i_face);
      if (face->at_boundary())
      {
        if (face->center()[2] > z_h - 1.e-6 ||
            face->center()[2] < z_0 + 1.e-6 || face->center()[1] < 1.e-6 ||
            face->center()[1] > 2. * M_PI - 1.e-6)
        {
          in_manager->BCs[i_face] = ModelEq::boundary_condition::periodic;
          in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 1);
        }
        else if (face->center()[0] > s_o - 1e-6)
        {
          in_manager->BCs[i_face] = ModelEq::boundary_condition::essential;
          in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 0);
        }
        else if (face->center()[0] < s_i + 1e-6)
        {
          in_manager->BCs[i_face] = ModelEq::boundary_condition::natural;
          in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 1);
        }
        else
        {
          // to make sure that all of the boundary conditions are taken care of:
          std::cout << face->center()[0] << std::endl;
          assert(false);
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

      problem_data_2<dim> data1;

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
        int solver_keys = nargil::solvers::solver_props::spd_matrix;
        int update_keys = nargil::solvers::solver_update_opts::update_mat |
                          nargil::solvers::solver_update_opts::update_rhs;
        //

        // PETScWrappers::SolverGMRES solver(solver_control, mpi_communicator);
        nargil::solvers::petsc_implicit_cg_solver<dim> solver1(
          solver_keys, dof_counter1, comm);

        // nargil::solvers::petsc_direct_solver<dim> solver1(solver_keys,
        // 						  dof_counter1,
        // 						  comm);
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

        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::fill_refn_vector, &dist_refn_vec);

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
          comm, &model_manager1.local_dof_handler, &global_sol_vec, "solution",
          "Temperature", "Heat_flow");
        viz_data1.time_step = 0;
        viz_data1.cycle = 0;

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
          "kappa_comps", "Temperature", "b_components");
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
      }
      //
      //
    }
    //

    PetscFinalize();
  }
};
