#include "../../include/models/model_manager.hpp"

template <int dim, int spacedim>
nargil::base_model_manager<dim, spacedim>::base_model_manager()
{
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::implicit_hybridized_model_manager<dim, spacedim>::
  implicit_hybridized_model_manager()
{
}

//
//

template <int dim, int spacedim>
template <typename BasisType, typename ModelEq>
void nargil::implicit_hybridized_model_manager<dim, spacedim>::
  form_dof_handlers(nargil::model<ModelEq, dim, spacedim> *in_model,
                    BasisType *in_basis)
{
  const dealii::FiniteElement<dim, spacedim> *local_fe =
    in_basis->get_local_fe();
  const dealii::FiniteElement<dim, spacedim> *trace_fe =
    in_basis->get_trace_fe();
  local_dof_handler.initialize(in_model->my_mesh->tria, *local_fe);
  trace_dof_handler.initialize(in_model->my_mesh->tria, *trace_fe);

  typedef typename BasisType::CellManagerType CellManagerType;
  auto active_owned_cell = in_model->all_owned_cells.begin();
  dealiiDoFCell i_local_cell = local_dof_handler.begin_active();
  dealiiDoFCell i_trace_cell = trace_dof_handler.begin_active();
  unsigned n_active_cells = in_model->my_mesh->tria.n_active_cells();

  for (unsigned i1 = 0; i1 < n_active_cells; ++i1)
  {
    if (i_local_cell->is_locally_owned())
    {
      ModelEq *i_cell = static_cast<ModelEq *>(active_owned_cell->get());
      auto i_manager = i_cell->template get_manager<CellManagerType>();
      i_manager->assign_dof_handler_cells(i_local_cell, i_trace_cell);
      ++active_owned_cell;
    }
    ++i_local_cell;
    ++i_trace_cell;
  }
}

//
//

template <int dim, int spacedim>
template <typename ModelEq, typename Func>
void nargil::implicit_hybridized_model_manager<dim, spacedim>::
  apply_func_to_owned_cells(model<ModelEq, dim, spacedim> *in_model, Func f)
{
  for (std::unique_ptr<cell<dim, spacedim> > &i_cell :
       in_model->all_owned_cells)
  {
    f(i_cell.get());
  }
}
