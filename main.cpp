#include <boost/dynamic_bitset.hpp>
#include <cstdio>
#include <fstream>
#include <vector>

#include <petsc.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include "include/mesh/mesh_handler.hpp"
#include "include/models/model.hpp"

/**
 * Just a sample problem
 */
template <int dim, int spacedim = dim> struct Problem
{
  /**
   * @brief generate_mesh
   */
  static void generate_mesh(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    std::vector<unsigned> repeats(dim, 10);
    dealii::Point<spacedim> point_1, point_2;
    point_1 = {-1.0, -1.0};
    point_2 = {1.0, 1.0};
    dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, repeats,
                                                      point_1, point_2, true);
  }

  /**
   * @brief dofs_on_nodes
   */
  boost::dynamic_bitset<> dofs_on_nodes()
  {
    boost::dynamic_bitset<> dof_names_on_nodes(1);
    dof_names_on_nodes[0] = 1;
    return dof_names_on_nodes;
  }

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
      std::vector<unsigned> repeats(dim, 10);
      dealii::Point<spacedim> point_1, point_2;
      point_1 = {-1.0, -1.0};
      point_2 = {1.0, 1.0};
      dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, repeats,
                                                        point_1, point_2, true);
    }
  };

  /**
   * @brief BCs_on_face
   */
  std::vector<nargil::boundary_condition> BCs_on_face()
  {
    std::vector<nargil::boundary_condition> BCs(
      1, nargil::boundary_condition::essential);
    return BCs;
  }
};

/**
 * @brief main
 */
int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, NULL);

  //
  {
    const MPI_Comm *const comm = &PETSC_COMM_WORLD;
    nargil::mesh<2> mesh1(*comm, 1, false);

    mesh1.generate_mesh(Problem<2>::generate_mesh);

    nargil::model_options::options options1 = (nargil::model_options::options)(
      nargil::model_options::implicit_time_integration |
      nargil::model_options::HDG_dof_numbering);
    nargil::model<nargil::diffusion_cell<2>, 2> model1(&mesh1, options1);

    nargil::bases::hdg_diffusion_polybasis<2> bases1(3, 4);
    model1.init_model_elements(&bases1);
    model1.count_globals();

    //    cell_container<2> cont1;
    //    cont1.set_dof_numbering(ModelOptions::implicit_type,
    //                            ModelOptions::hybridized_DG);

    //
    // We can also use a functor to generate the mesh.
    //
    // h_mesh1.generate_mesh(problem<2>::grid_gen2());
  }
  //

  PetscFinalize();
}
