#include "../../include/models//hybridized_DG.hpp"

template <int dim, int spacedim>
nargil::HDG_dof_numbering<dim, spacedim>::HDG_dof_numbering()
  : dof_numbering<dim, spacedim>()
{
}

template <int dim, int spacedim>
void nargil::HDG_dof_numbering<dim, spacedim>::init_mesh_containers()
{
}

template <int dim, int spacedim>
void nargil::HDG_dof_numbering<dim, spacedim>::set_boundary_indicator()
{
}

template <int dim, int spacedim>
void nargil::HDG_dof_numbering<dim, spacedim>::count_globals()
{
}
