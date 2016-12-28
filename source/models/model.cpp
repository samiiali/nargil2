#include "../../include/models/model.hpp"

template <int dim, int spacedim>
nargil::dof_numbering<dim, spacedim>::dof_numbering()
{
}

template <int dim, int spacedim>
nargil::dof_numbering<dim, spacedim>::~dof_numbering()
{
}

//
//
//
//
//
template <int dim, int spacedim>
nargil::implicit_HDG_dof_numbering<dim, spacedim>::implicit_HDG_dof_numbering()
  : dof_numbering<dim, spacedim>()
{
}

template <int dim, int spacedim>
nargil::implicit_HDG_dof_numbering<dim, spacedim>::~implicit_HDG_dof_numbering()
{
}

//
//
//
//
//
nargil::BaseModel::BaseModel() {}

nargil::BaseModel::~BaseModel() {}

//
//
//
//
//
template <int dim, int spacedim>
nargil::Model<dim, spacedim>::Model(Mesh<dim, spacedim> *const mesh_)
  : mesh(mesh_)
{
}

template <int dim, int spacedim> nargil::Model<dim, spacedim>::~Model() {}

template <int dim, int spacedim>
void nargil::Model<dim, spacedim>::set_dof_numbering_type(
  const ModelOptions::Options options)
{
  if (options == ModelOptions::implicit_time_integration |
      ModelOptions::HDG_dof_numbering)
  {
    dof_counter =
      std::move(std::unique_ptr<implicit_HDG_dof_numbering<dim, spacedim> >(
        new implicit_HDG_dof_numbering<dim, spacedim>()));
  }
}

template <int dim, int spacedim>
template <typename ModelEq>
void nargil::Model<dim, spacedim>::assign_model_features()
{
  all_owned_cells.reserve(mesh->tria.n_locally_owned_active_cells());
  unsigned n_owned_cell = 0;
  for (dealii_cell_type &&cell : mesh->tria.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      all_owned_cells.push_back(Cell<dim, spacedim>::template create<ModelEq>(
        cell, n_owned_cell, this));
      ++n_owned_cell;
    }
  }
}
