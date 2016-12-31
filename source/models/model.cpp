#include "../../include/models/model.hpp"

//
//
//
//
//
nargil::base_model::base_model() {}

nargil::base_model::~base_model() {}

//
//
//
//
//
template <typename ModelEq, int dim, int spacedim>
nargil::model<ModelEq, dim, spacedim>::model(
  mesh<dim, spacedim> *const mesh_, const model_options::options options)
  : my_mesh(mesh_),
    my_dof_counter(new dof_numbering<dim, spacedim>()),
    my_opts(options)
{
  set_dof_numbering_type();
}

template <typename ModelEq, int dim, int spacedim>
nargil::model<ModelEq, dim, spacedim>::~model()
{
}

template <typename ModelEq, int dim, int spacedim>
void nargil::model<ModelEq, dim, spacedim>::set_dof_numbering_type()
{
  if (my_opts == implicit_HDG_dof_numbering<dim, spacedim>::options())
  {
    my_dof_counter =
      std::move(std::unique_ptr<implicit_HDG_dof_numbering<dim, spacedim> >(
        new implicit_HDG_dof_numbering<dim, spacedim>()));
  }
  else
  {
    assert(false && "The options for the dof numbering were not recognized.");
  }
}

template <typename ModelEq, int dim, int spacedim>
template <typename BasisType>
void nargil::model<ModelEq, dim, spacedim>::init_model_elements(
  BasisType *basis)
{
  all_owned_cells.reserve(my_mesh->tria.n_locally_owned_active_cells());
  unsigned n_owned_cell = 0;
  for (dealii_cell_type &&i_cell : my_mesh->tria.active_cell_iterators())
  {
    if (i_cell->is_locally_owned())
    {
      all_owned_cells.push_back(cell<dim, spacedim>::template create<ModelEq>(
        i_cell, n_owned_cell, basis, this));
      ++n_owned_cell;
    }
  }
}

template <typename ModelEq, int dim, int spacedim>
template <typename Func>
void nargil::model<ModelEq, dim, spacedim>::set_boundary_indicator(Func f)
{
  for (dealii_cell_type &&cell : all_owned_cells)
  {
    static_cast<ModelEq *>(cell.get())->assign_BCs(f);
  }
}

template <typename ModelEq, int dim, int spacedim>
void nargil::model<ModelEq, dim, spacedim>::count_globals()
{
  if (my_opts == implicit_HDG_dof_numbering<dim, spacedim>::options())
  {
    static_cast<implicit_HDG_dof_numbering<dim, spacedim> *>(
      my_dof_counter.get())
      ->template count_globals<model<ModelEq, dim, spacedim>, ModelEq>(this);
  }
  else
  {
    assert(false && "The options for the dof numbering were not recognized.");
  }
}
