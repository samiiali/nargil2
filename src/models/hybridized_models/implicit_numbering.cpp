#include "implicit_numbering.hpp"

template <int dim>
hybridized_dof_numbering<dim>::hybridized_dof_numbering()
  : generic_dof_numbering<dim>()
{
}

template <int dim> void hybridized_dof_numbering<dim>::init_mesh_containers() {}

template <int dim> void hybridized_dof_numbering<dim>::set_boundary_indicator()
{
}

template <int dim> void hybridized_dof_numbering<dim>::count_globals() {}
