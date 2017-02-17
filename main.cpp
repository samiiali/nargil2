#include <cstdio>
#include <fstream>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include <petsc.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/base/index_set.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/parallel_vector.h>

#include "include/elements/diffusion.hpp"
#include "include/mesh/mesh_handler.hpp"
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
template <int dim, int spacedim = dim> struct Problem
{
  typedef nargil::diffusion<2> ModelEq;
  typedef nargil::model<ModelEq, 2> ModelType;
  typedef ModelEq::hdg_polybasis BasisType;
  typedef nargil::diffusion<2>::hdg_manager CellManagerType;

  /**
   * @brief generate_mesh
   */
  static void generate_mesh(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    std::vector<unsigned> repeats(dim, 4);
    dealii::Point<spacedim> point_1, point_2;
    point_1 = {-1.0, -1.0};
    point_2 = {1.0, 1.0};
    dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, repeats,
                                                      point_1, point_2, true);
  }

  /**
   * @brief adaptive_mesh_gen
   */
  static void adaptive_mesh_gen(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    std::vector<unsigned> repeats(dim, 4);
    dealii::Point<spacedim> point_1, point_2;
    point_1 = {-1.0, -1.0};
    point_2 = {1.0, 1.0};
    dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, repeats,
                                                      point_1, point_2, true);
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
  }

  /**
   * @brief solver_checker_mesh
   */
  static void solver_checker_mesh(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    std::vector<unsigned> repeats(dim, 2);
    dealii::Point<spacedim> point_1, point_2;
    point_1 = {-1.0, -1.0};
    point_2 = {1.0, 1.0};
    dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, repeats,
                                                      point_1, point_2, true);
    typedef typename nargil::mesh<dim, spacedim>::dealiiTriCell dealiiTriCell;
    dealii::Point<dim> refn_point1(0.5, 0.5);
    for (dealiiTriCell &&i_cell : the_mesh.active_cell_iterators())
    {
      if (i_cell->is_locally_owned() && i_cell->point_inside(refn_point1))
      {
        i_cell->set_refine_flag();
      }
    }
    the_mesh.execute_coarsening_and_refinement();
  }

  /**
   * @brief dofs_on_nodes
   */
  static void assign_BCs(nargil::cell<dim, spacedim> *in_cell)
  {
    unsigned n_dof_per_face = BasisType::get_n_dofs_per_face();
    auto i_manager =
      static_cast<ModelEq *>(in_cell)
        ->template get_manager<typename BasisType::CellManagerType>();
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

  static double exact_uhat_func2(dealii::Point<spacedim> p)
  {
    return sin(p[0]);
  }

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

  //
  // This is a functor, which basically does the same thing as above.
  // Since, we do not need the state of the grid_gen, we really do not
  // need a functor here. I have just implemented it for later reference.
  //
  /**
   * @brief The grid_gen2 struct
   */
  struct grid_gen2
  {
    /**
     * @brief operator ()
     */
    void operator()(
      dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
    {
      std::vector<unsigned> repeats(dim, 4);
      dealii::Point<spacedim> point_1, point_2;
      point_1 = {-1.0, -1.0};
      point_2 = {1.0, 1.0};
      dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, repeats,
                                                        point_1, point_2, true);
    }
  };
};

//
//
//
//
//

//
//
//
//
//

/**
 * @brief main
 */
int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, NULL);
  dealii::MultithreadInfo::set_thread_limit(1);

  //
  {
    const MPI_Comm *const comm = &PETSC_COMM_WORLD;
    int comm_rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &comm_rank);

    nargil::mesh<2> mesh1(*comm, 1, false);

    //    mesh1.generate_mesh(Problem<2>::adaptive_mesh_gen);
    mesh1.generate_mesh(Problem<2>::generate_mesh);

    // We can also use a functor to generate the mesh.
    // h_mesh1.generate_mesh(problem<2>::grid_gen2());

    Problem<2>::ModelType model1(mesh1);

    Problem<2>::BasisType basis1(3, 4);
    model1.init_model_elements(&basis1);
    model1.assign_BCs<Problem<2>::BasisType>(Problem<2>::assign_BCs);

    nargil::implicit_hybridized_numbering<2> dof_counter1;
    dof_counter1.count_globals<Problem<2>::BasisType>(&model1);

    nargil::simple_implicit_solver<2> solver1(dof_counter1);

    nargil::implicit_hybridized_model_manager<2> model_manager1;
    model_manager1.form_dof_handlers(&model1, &basis1);

    //
    //

    auto exact_uhat_func = [](const dealii::Point<2> &p) { return sin(p[0]); };
    model_manager1.apply_func_to_owned_cells(
      &model1, Problem<2>::CellManagerType::run_interpolate_to_trace,
      exact_uhat_func);

    nargil::distributed_vector<2> exact_sol_vec(
      model_manager1.local_dof_handler, MPI_COMM_WORLD);
    auto exact_local_func = [](const dealii::Point<2> &p) {
      std::vector<double> out(3);
      out[0] = sin(p[0]) + cos(p[1]);
      out[1] = -cos(p[0]);
      out[2] = sin(p[1]);
      return out;
    };
    model_manager1.apply_func_to_owned_cells(
      &model1, Problem<2>::CellManagerType::run_interpolate_to_interior,
      exact_local_func, &exact_sol_vec);

    LA::MPI::Vector global_nodal_vec;
    exact_sol_vec.copy_to_global_vec(global_nodal_vec);
    /*
    Problem<2>::CellManagerType::visualize_results(
      model_manager1.local_dof_handler, global_nodal_vec, 0.0);
    */

    //    Problem<2>::exact_sol_func exact_sol;
    //    LA::MPI::Vector exact_sol_vec2(
    //      model_manager1.local_dof_handler.locally_owned_dofs(),
    //      PETSC_COMM_WORLD);
    //    dealii::VectorTools::interpolate(model_manager1.local_dof_handler,
    //                                     exact_sol, exact_sol_vec2);
    //    Problem<2>::CellManagerType::visualize_results(
    //      model_manager1.local_dof_handler, exact_sol_vec2, 0.0);
    //    model_manager1.apply_func_to_owned_cells(
    //      &model1,
    //      Problem<2>::CellManagerType::run_assign_exact_local_unkns,
    //      exact_sol_vec);

    model_manager1.apply_func_to_owned_cells(
      &model1, Problem<2>::CellManagerType::run_compute_local_unkns);

    std::vector<double> sum_of_L2_errors(2, 0);
    model_manager1.apply_func_to_owned_cells(
      &model1, Problem<2>::CellManagerType::run_compute_errors,
      &sum_of_L2_errors);

    //
    //

    std::vector<int> row({0, 1, 2, 3});
    Eigen::MatrixXd A_mat(4, 4);
    A_mat << 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4;
    Eigen::VectorXd b_vec(4);
    Eigen::VectorXd x_vec;
    b_vec << 2, 2, 6, 8;
    solver1.push_to_global_mat(row.data(), row.data(), A_mat,
                               nargil::insert_mode::ins_vals);
    solver1.push_to_rhs_vec(row.data(), b_vec, nargil::insert_mode::ins_vals);
    solver1.finish_assemble();
    solver1.form_factors();
    solver1.solve_system(x_vec);
  }
  //

  PetscFinalize();
}
