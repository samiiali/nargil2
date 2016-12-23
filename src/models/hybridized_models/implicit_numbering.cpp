#include "implicit_numbering.hpp"

template <int dim, int spacedim>
hybridized_dof_numbering<dim, spacedim>::hybridized_dof_numbering()
  : dof_numbering<dim, spacedim>()
{
}

template <int dim, int spacedim>
void hybridized_dof_numbering<dim, spacedim>::init_mesh_containers()
{
}

template <int dim, int spacedim>
void hybridized_dof_numbering<dim, spacedim>::set_boundary_indicator()
{
}

template <int dim, int spacedim>
void hybridized_dof_numbering<dim, spacedim>::count_globals()
{
}
