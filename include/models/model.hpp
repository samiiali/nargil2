#include <assert.h>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#ifndef GENERIC_DOF_NUM_HPP
#define GENERIC_DOF_NUM_HPP

#include "../elements/cell.hpp"
#include "../mesh/mesh_handler.hpp"
#include "dof_numbering.hpp"

namespace nargil
{
//
//
//
<<<<<<< HEAD
=======
// Forward declerations
template <int dim, int spacedim> struct mesh;
template <int dim, int spacedim> struct dof_numbering;
template <int dim, int spacedim> struct implicit_hybridized_dof_numbering;

//
//
//
>>>>>>> aa2b09de15e38bbfa49319eaad6375082ee8d2ab
//
/**
 * This is an abstract model that all other models will be based on it.
 * Thus it palys the rule of an interface for the developers (and not the
 * users).
 */
struct base_model
{
  base_model();
  ~base_model();
};

//
//
//
//
/**
 * The model class contains is actually used to solve a model problem.
 */
template <typename ModelEq, int dim, int spacedim = dim>
struct model : public base_model
{
  //
  //
  /**
   * @brief This typename is used to count and iterate over the deal.II cells.
   */
  typedef dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim> >
    dealii_cell_type;

  //
  //
  /**
   * @brief Constructor of the class.
   */
  model(mesh<dim, spacedim> *const);

  //
  //
  /**
   * @brief Destructor of the class.
   */
  ~model();

  //
  //
  /**
   * This function sets how we should choose the Model::dof_counter.
   * This is called in the constructor.
   *
   * @todo We should assert if dof_counter is set here, before using
   * it anywhere else.
   */
  void set_dof_numbering_type();

  //
  //
  /**
   * @brief Assigns the dof_numbering type. The user first creates a special
   * type of dof_numbering and then assign it to the model.
   * @todo Adding the capability to have more than one dof_numbering and being
   * able to change between them.
   */
  template <typename DOF_NUMBERING>
  void set_dof_numbering(std::unique_ptr<DOF_NUMBERING>);

  //
  //
  /**
   * This function initiates the member Model::all_owned_cells, based on the
   * equation that we want to solve. This is called in the constructor of the
   * class.
   */
  template <typename BasisType> void init_model_elements(BasisType *);

  //
  //
  /**
   * @brief Here, we set the boundary indicator of each face on the boundary.
   */
  template <typename Func> void set_boundary_indicator(Func f);

  //
  //
  /**
   * @brief Here we count the global DOFs of the mesh.
   */
  void count_globals();

  //
  //
  /**
   * @brief This function frees the memory used by the model.
   */
  void free_containers();

  //
  //
  /**
   * @brief This is a pointer to the mesh that the model is working on.
   */
  mesh<dim, spacedim> *my_mesh;

  //
  //
  /**
   * This is a std::vector containing all of the Cell classes in the model.
   */
  std::vector<std::unique_ptr<cell<dim, spacedim> > > all_owned_cells;

  //
  //
  /**
   * This is an instance of the dof_numbering class.
   */
  std::unique_ptr<dof_numbering<dim, spacedim> > my_dof_counter;

  //
  //
  /**
   * @brief my_opts
   */
  model_options::options my_opts;
};
}

#include "../../source/models/model.cpp"

#include "dof_numbering.hpp"

#endif
