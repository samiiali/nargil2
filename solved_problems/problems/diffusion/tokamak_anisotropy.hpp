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
struct problem_data : public nargil::diffusion<dim, spacedim>::data
{
  /**
   * @brief pi
   */
  const double pi = M_PI;
  const double epsinv = 1.0e3;
  const double r_i = 0.7;
  const double r_o = 1.0;
  /**
   * @brief Constructor.
   */
  problem_data() : nargil::diffusion<dim, spacedim>::data() {}

  /**
   * @brief rhs_func.
   */
  virtual double rhs_func(const dealii::Point<spacedim> &p)
  {
    return 0.0;

    //-0.25 * cos(2 * p[1]) * sin(p[0]) *
    //   (3 * sin(p[2]) - 74 * sin(3 * p[2]) + 15 * sin(5 * p[2]));
  }

  /**
   * @brief gD_func.
   */
  virtual double gD_func(const dealii::Point<spacedim> &p)
  {
    double temperature = 1.0;
    if (sqrt(p[0] * p[0] + p[1] * p[1]) > 1.0 - 1.e-4)
      temperature = 0.0;
    return temperature;
  }

  /**
   * @brief gN_func.
   */
  virtual dealii::Tensor<1, dim> gN_func(const dealii::Point<spacedim> &p)
  {
    return dealii::Tensor<1, dim>({0, 0, 0});

    //     ({-cos(p[0]) * cos(2 * p[1]) * sin(3 * p[2]),
    //        2 * sin(p[0]) * sin(2 * p[1]) * sin(3 * p[2]),
    //        -3 * cos(2 * p[1]) * cos(3 * p[2]) * sin(p[0]) *
    //          (1 + sin(p[2]) * sin(p[2]))});
  }

  /**
   * @brief exact_u
   */
  virtual double exact_u(const dealii::Point<spacedim> &p)
  {
    return sin(p[0]) * cos(2. * p[1]) * sin(3. * p[2]);
  }

  /**
   * @brief exact_q
   */
  virtual dealii::Tensor<1, dim> exact_q(const dealii::Point<spacedim> &p)
  {
    return dealii::Tensor<1, dim>(
      {-cos(p[0]) * cos(2 * p[1]) * sin(3 * p[2]),
       2 * sin(p[0]) * sin(2 * p[1]) * sin(3 * p[2]),
       -3 * cos(2 * p[1]) * cos(3 * p[2]) * sin(p[0]) *
         (1 + sin(p[2]) * sin(p[2]))});
  }

  /**
   *
   */
  virtual dealii::Tensor<2, dim> kappa_inv(const dealii::Point<spacedim> &p)
  {

    double r = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    double z = p[2];
    double theta;

    if (p[0] != 0.0)
    {
      theta = std::atan(p[1] / p[0]);
    }
    else if (p[1] > 0.0)
    {
      theta = pi / 2.0;
    }
    else if (p[1] < 0)
    {
      theta = 3.0 * pi / 2.0;
    }

    double psitilde = 0.002;
    double psishape = (r * r) * (1. - r) * (1. - r);
    double psishapep = 2.0 * r * (1. - r) * (1. - r) -
                       2. * r * r * (1. - r); // diff(psishape, x);
    double psi32 = std::cos(3.0 * theta - 2.0 * z);
    double psi43 = std::cos(4.0 * theta - 3.0 * z);
    double psip32 = -3.0 * std::sin(3.0 * theta - 2.0 * z);
    double psip43 = -4.0 * std::sin(4.0 * theta - 3.0 * z);
    double psipert = psitilde * (psi32 + psi43);
    double psipertp = psitilde * (psip32 + psip43);
    double qsafety = 0.2 * std::exp(r / 0.3);
    double br = psishape * psipertp;
    double btheta = -psishapep * psipert + 1.0 / qsafety;
    double bz = 1.0;

    dealii::Tensor<2, dim> result;
    // result[0][0] = 1.0 + (epsinv - 1.) * br * br;
    // result[0][1] = (epsinv - 1.) * br * btheta;
    // result[0][2] = (epsinv - 1.) * br * bz;
    // result[1][0] = result[0][1];
    // result[1][1] = 1.0 / (r * r) + (epsinv - 1.) * btheta * btheta;
    // result[1][2] = (epsinv - 1) * btheta * bz;
    // result[2][0] = result[0][2];
    // result[2][1] = result[1][2];
    // result[2][2] = 1. + (epsinv - 1.) * bz * bz;
    // result[2][2] = 1. / (1. + sin(p[2]) * sin(p[2]));
    result[0][0] = 1.0;
    result[1][1] = 1.0;
    result[2][2] = 1.0;
    return dealii::invert(result);
  }

  /**
   *
   */
  virtual double tau(const dealii::Point<spacedim> &)
  {
    //
    return 1.0e6;
    //
  }
};

/**
 * Just a sample problem
 */
template <int dim, int spacedim = dim> struct Problem1
{
  typedef nargil::diffusion<dim> ModelEq;
  typedef nargil::model<ModelEq, dim> ModelType;
  typedef typename ModelEq::hdg_polybasis BasisType;
  typedef typename nargil::diffusion<dim>::template hdg_manager<BasisType>
    CellManagerType;

  /**
   * @brief adaptive_mesh_gen
   */
  static void mesh_gen(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    // r_o , r_i are redefined here.
    double r_i = 0.7;
    double r_o = 1.0;
    dealii::CylindricalManifold<dim> manifold1(2);
    dealii::GridGenerator::cylinder_shell(the_mesh, 2 * M_PI, r_i, r_o, 15, 1);

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
          if (face_center[2] > 2 * M_PI - 1.e-4)
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
      dealii::Tensor<1, dim>({0., 0., 2 * M_PI}));
    the_mesh.add_periodicity(periodic_faces);

    the_mesh.set_all_manifold_ids(0);
    the_mesh.set_manifold(0, manifold1);
    the_mesh.refine_global(5);
    the_mesh.set_manifold(0);
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
        if (face->center()[2] > 2. * M_PI - 1.E-4 || face->center()[2] < 1.E-4)
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

      problem_data<dim> data1;

      mesh1.generate_mesh(mesh_gen);
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
          &model1, CellManagerType::assign_BCs, assign_BCs);
        model_manager1.apply_on_ghost_cells(
          &model1, CellManagerType::assign_BCs, assign_BCs);
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
        nargil::solvers::petsc_implicit_cg_solver<dim> solver1(
          solver_keys, dof_counter1, comm);
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

        CellManagerType::visualize_results(model_manager1.local_dof_handler,
                                           global_sol_vec, i_cycle);
        model_manager1.apply_on_owned_cells(
          &model1, CellManagerType::interpolate_to_interior); // exact
                                                              // interpolated to
                                                              // u and q
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
