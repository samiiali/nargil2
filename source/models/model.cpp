#include "../../include/models/model.hpp"

//
//
//
//
//
nargil::base_model::base_model() {}

//
//

nargil::base_model::~base_model() {}

//
//
//
//
//
template <typename ModelEq, int dim, int spacedim>
nargil::model<ModelEq, dim, spacedim>::model(const mesh<dim, spacedim> &in_mesh)
  : my_mesh(&in_mesh)
{
}

//
//

template <typename ModelEq, int dim, int spacedim>
nargil::model<ModelEq, dim, spacedim>::~model()
{
}

//
//

template <typename ModelEq, int dim, int spacedim>
template <typename BasisType>
void nargil::model<ModelEq, dim, spacedim>::init_model_elements(
  const BasisType &basis)
{
  all_owned_cells.reserve(my_mesh->n_owned_cell);
  all_ghost_cells.reserve(my_mesh->n_ghost_cell);
  unsigned i_owned_cell = 0;
  unsigned i_ghost_cell = 0;
  for (dealii_cell_type &&i_cell : my_mesh->tria.active_cell_iterators())
  {
    if (i_cell->is_locally_owned())
    {
      all_owned_cells.push_back(cell<dim, spacedim>::template create<ModelEq>(
        i_cell, i_owned_cell, basis, this));
      ++i_owned_cell;
    }
    if (i_cell->is_ghost())
    {
      all_ghost_cells.push_back(cell<dim, spacedim>::template create<ModelEq>(
        i_cell, i_ghost_cell, basis, this));
      ++i_ghost_cell;
    }
  }
}

//
//

template <typename ModelEq, int dim, int spacedim>
template <typename BasisType, typename Func>
void nargil::model<ModelEq, dim, spacedim>::assign_BCs(Func f)
{
  for (auto &&i_cell : all_owned_cells)
    static_cast<ModelEq *>(i_cell.get())->assign_BCs<BasisType>(f);
  // Applying the BCs on ghost cells.
  for (auto &&i_cell : all_ghost_cells)
    static_cast<ModelEq *>(i_cell.get())->assign_BCs<BasisType>(f);
}

//
//

template <typename ModelEq, int dim, int spacedim>
template <typename CellManagerType, typename InputType>
CellManagerType *nargil::model<ModelEq, dim, spacedim>::get_owned_cell_manager(
  const InputType &cell_id) const
{
  int num_id = my_mesh->cell_id_to_num_finder(cell_id, true);
  CellManagerType *i_manager =
    static_cast<ModelEq *>(all_owned_cells[num_id].get())
      ->template get_manager<CellManagerType>();
  return i_manager;
}

//
//

template <typename ModelEq, int dim, int spacedim>
template <typename CellManagerType, typename InputType>
CellManagerType *nargil::model<ModelEq, dim, spacedim>::get_ghost_cell_manager(
  const InputType &cell_id) const
{
  int num_id = my_mesh->cell_id_to_num_finder(cell_id, false);
  CellManagerType *i_manager =
    static_cast<ModelEq *>(all_ghost_cells[num_id].get())
      ->template get_manager<CellManagerType>();
  return i_manager;
}
