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
  typedef nargil::diffusion<2> ModelEq;
  typedef nargil::model<ModelEq, 2> ModelType;
  typedef ModelEq::hdg_polybasis BasisType;
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
  static void assign_BCs(nargil::cell<dim, spacedim> *in_cell)
  {
    unsigned n_dof_per_face = BasisType::get_n_dofs_per_face();
    auto i_manager =
      static_cast<ModelEq *>(in_cell)
        ->template get_manager<typename BasisType::CellManagerType>();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      auto &&face = i_manager->my_cell->dealii_cell->face(i_face);
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

    Problem<2>::ModelType model1(mesh1);

    Problem<2>::BasisType basis1(3, 4);
    model1.init_model_elements(basis1);
    model1.assign_BCs<Problem<2>::BasisType>(Problem<2>::assign_BCs);

    nargil::implicit_hybridized_numbering<2> dof_counter1;
    dof_counter1.count_globals<Problem<2>::BasisType>(&model1);

    //
    // We can also use a functor to generate the mesh.
    //
    // h_mesh1.generate_mesh(problem<2>::grid_gen2());
  }
  //

  PetscFinalize();
}
