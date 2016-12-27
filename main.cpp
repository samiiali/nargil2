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

template <int dim, int spacedim = dim> struct Problem
{
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
  struct grid_gen2
  {
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

  std::vector<nargil::BC> BCs_on_face()
  {
    std::vector<nargil::BC> BCs(1, nargil::BC::essential);
    return BCs;
  }
};

int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, NULL);

  //
  {
    const MPI_Comm *const comm = &PETSC_COMM_WORLD;
    nargil::Mesh<2> mesh1(*comm, 1, false);

    mesh1.generate_mesh(Problem<2>::generate_mesh);

    nargil::Model<nargil::Diffusion<2>, 2> model1(&mesh1);
    model1.init_mesh_containers();

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
