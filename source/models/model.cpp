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
nargil::model<ModelEq, dim, spacedim>::model(mesh<dim, spacedim> *const mesh_)
  : my_mesh(mesh_)
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
template <typename T>
void nargil::model<ModelEq, dim, spacedim>::set_dof_numbering(
  std::unique_ptr<T> dof_counter)
{
  my_opts = T::get_options();
  my_dof_counter = std::move(dof_counter);
}

//
//

template <typename ModelEq, int dim, int spacedim>
template <typename BasisType>
void nargil::model<ModelEq, dim, spacedim>::init_model_elements(
  BasisType *basis)
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
template <typename Func>
void nargil::model<ModelEq, dim, spacedim>::assign_BCs(Func f)
{
  for (auto &&i_cell : all_owned_cells)
    static_cast<ModelEq *>(i_cell.get())->assign_BCs(f);
  // Applying the BCs on ghost cells.
  for (auto &&i_cell : all_ghost_cells)
    static_cast<ModelEq *>(i_cell.get())->assign_BCs(f);
}

//
//

template <typename ModelEq, int dim, int spacedim>
void nargil::model<ModelEq, dim, spacedim>::count_globals()
{
  if (my_opts ==
      implicit_hybridized_dof_numbering<dim, spacedim>::get_options())
  {
    static_cast<implicit_hybridized_dof_numbering<dim, spacedim> *>(
      my_dof_counter.get())
      ->template count_globals<ModelEq>(this);
  }
  else
  {
    assert(false && "The options for the dof numbering were not recognized.");
  }
}
