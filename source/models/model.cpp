#include "../../include/models/model.hpp"

template <int dim, int spacedim>
nargil::dof_numbering<dim, spacedim>::dof_numbering()
{
}

template <int dim, int spacedim>
nargil::dof_numbering<dim, spacedim>::~dof_numbering()
{
}

template <int dim, int spacedim>
unsigned nargil::dof_numbering<dim, spacedim>::get_global_mat_block_size()
{
  return 0;
  //  unsigned poly_order = manager->poly_order;
  //  unsigned n_face_basis = pow(poly_order, dim - 1);
  //  return n_face_basis * CellType<dim>::get_num_dofs_per_node();
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
template <typename ModelEq, int dim, int spacedim>
nargil::Model<ModelEq, dim, spacedim>::Model(Mesh<dim, spacedim> *const mesh_)
  : mesh(mesh_)
{
}

template <typename ModelEq, int dim, int spacedim>
nargil::Model<ModelEq, dim, spacedim>::~Model()
{
}

template <typename ModelEq, int dim, int spacedim>
void nargil::Model<ModelEq, dim, spacedim>::set_dof_numbering_type(
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

template <typename ModelEq, int dim, int spacedim>
void nargil::Model<ModelEq, dim, spacedim>::init_mesh_containers()
{
  std::cout << mesh->dealii_mesh.n_locally_owned_active_cells() << std::endl;
  all_owned_cells.reserve(mesh->dealii_mesh.n_locally_owned_active_cells());
  //  unsigned n_cell = 0;
  //  this->manager->n_ghost_cell = 0;
  //  this->manager->n_owned_cell = 0;
  //  for (dealiiCell &&cell : this->DoF_H_System.active_cell_iterators())
  //  {
  //    if (cell->is_locally_owned())
  //    {
  //      all_owned_cells.push_back(GenericCell<dim>::template
  //      make_cell<CellType>(
  //        cell, this->manager->n_owned_cell, poly_order, this));
  //      this->manager->cell_ID_to_num[all_owned_cells.back()->cell_id] =
  //        this->manager->n_owned_cell;
  //      ++this->manager->n_owned_cell;
  //    }
  //    if (cell->is_ghost())
  //      ++this->manager->n_ghost_cell;
  //    ++n_cell;
  //  }
}
