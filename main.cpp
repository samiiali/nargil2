#include <cstdio>
#include <fstream>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include <petsc.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include "include/elements/diffusion.hpp"
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
  static void assign_BCs(
    typename nargil::diffusion<dim, spacedim>::hdg_worker *const worker)
  {
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      auto &&face = worker->my_cell->dealii_cell->face(i_face);
      if (face->at_boundary())
      {
        if (fabs(face->center()[0] > 1 - 1.E-4))
        {
          worker->BCs[i_face] = nargil::boundary_condition::essential;
          worker->dof_names_on_faces[i_face].resize(1, 0);
        }
        else
        {
          worker->BCs[i_face] = nargil::boundary_condition::essential;
          worker->dof_names_on_faces[i_face].resize(1, 0);
        }
      }
      else
      {
        worker->dof_names_on_faces[i_face].resize(1, 1);
      }
    }
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

    nargil::model<nargil::diffusion<2>, 2> model1(&mesh1);

    std::unique_ptr<nargil::implicit_hybridized_dof_numbering<2> >
      p_dof_counter(new nargil::implicit_hybridized_dof_numbering<2>());

    model1.set_dof_numbering(std::move(p_dof_counter));

    nargil::diffusion<2>::hdg_polybasis bases1(3, 4);
    model1.init_model_elements(&bases1);
    model1.assign_BCs(Problem<2>::assign_BCs);
    model1.count_globals();

    //
    // We can also use a functor to generate the mesh.
    //
    // h_mesh1.generate_mesh(problem<2>::grid_gen2());
  }
  //

  PetscFinalize();
}
