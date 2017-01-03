#include "../../include/models/dof_numbering.hpp"

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
nargil::implicit_hybridized_dof_numbering<dim, spacedim>::implicit_hybridized_dof_numbering()
  : dof_numbering<dim, spacedim>()
{
  std::cout << "constructor of implicit_HDG_dof_numbering" << std::endl;
}

template <int dim, int spacedim>
nargil::implicit_hybridized_dof_numbering<dim, spacedim>::~implicit_hybridized_dof_numbering()
{
}

template <int dim, int spacedim>
nargil::model_options::options
nargil::implicit_hybridized_dof_numbering<dim, spacedim>::get_options()
{
  return (model_options::options)(model_options::implicit_time_integration |
                                  model_options::hybridized_dof_numbering);
}

template <int dim, int spacedim>
template <typename ModelType, typename ModelEq>
void nargil::implicit_hybridized_dof_numbering<dim, spacedim>::count_globals(
  ModelType *in_model)
{
  static_cast<ModelEq *>(in_model->all_owned_cells[0].get())
    ->get_relevant_dofs_count(0);
}
