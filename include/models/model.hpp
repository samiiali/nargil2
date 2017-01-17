#include <assert.h>
#include <functional>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

#ifndef GENERIC_DOF_NUM_HPP
#define GENERIC_DOF_NUM_HPP

#include "../elements/cell.hpp"
#include "../mesh/mesh_handler.hpp"
#include "dof_counter.hpp"

namespace nargil
{
//
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
    dealiiCell;

  //
  //
  /**
   * @brief Constructor of the class.
   */
  model(const mesh<dim, spacedim> &);

  //
  //
  /**
   * @brief Destructor of the class.
   */
  ~model();

  //
  //
  /**
   * This function initiates the member Model::all_owned_cells, based on the
   * equation that we want to solve. This is called in the constructor of the
   * class.
   */
  template <typename BasisType> void init_model_elements(const BasisType &);

  //
  //
  /**
   * @brief Here, we set the boundary indicator of each face on the boundary.
   */
  template <typename BasisType, typename Func> void assign_BCs(Func f);

  //
  //
  /**
   * @brief Returns the manager of the cell. The input is either a
   * dealii cell or a cellID of a dealii cell.
   */
  template <typename CellManagerType, typename InputType>
  CellManagerType *get_owned_cell_manager(const InputType &) const;

  //
  //
  /**
   * @brief Returns the manager of the cell. The input is either a
   * dealii cell or a cellID of a dealii cell.
   */
  template <typename BasisType, typename InputType>
  const BasisType *get_owned_cell_basis(const InputType &) const;

  //
  //
  /**
   * @brief Returns the manager of the cell.
   */
  template <typename CellManagerType, typename InputType>
  CellManagerType *get_ghost_cell_manager(const InputType &) const;

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
  const mesh<dim, spacedim> *my_mesh;

  //
  //
  /**
   * This is a std::vector containing all of the Cell classes in the model.
   */
  std::vector<std::unique_ptr<cell<dim, spacedim> > > all_owned_cells;

  //
  //
  /**
   * This is a std::vector containing all of the ghost cells in the model.
   */
  std::vector<std::unique_ptr<cell<dim, spacedim> > > all_ghost_cells;

  //
  //
  /**
   * This is an instance of the dof_counter class.
   */
  std::unique_ptr<dof_counter<dim, spacedim> > my_dof_counter;
};
}

#include "../../source/models/model.cpp"

#include "dof_counter.hpp"

#endif
