#include "../../include/misc/utils.hpp"

void nargil::Tokenize(const std::string &str_in,
                      std::vector<std::string> &tokens,
                      const std::string &delimiters = " ")
{
  auto lastPos = str_in.find_first_not_of(delimiters, 0);
  auto pos = str_in.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos)
  {
    tokens.push_back(str_in.substr(lastPos, pos - lastPos));
    lastPos = str_in.find_first_not_of(delimiters, pos);
    pos = str_in.find_first_of(delimiters, lastPos);
  }
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::distributed_vector<dim, spacedim>::distributed_vector(
  const dealii::DoFHandler<dim, spacedim> &dof_handler, const MPI_Comm &comm)
  : my_dof_handler(&dof_handler),
    idx_set(my_dof_handler->locally_owned_dofs()),
    my_comm(&comm),
    local_nodal_vec(idx_set, *my_comm)
{
  idx_set.fill_index_vector(idx_vec);
}

//
//

template <int dim, int spacedim>
void nargil::distributed_vector<dim, spacedim>::assemble(const unsigned &idx,
                                                         const double &val)
{
  local_nodal_vec[idx_vec[idx]] = val;
}

//
//

template <int dim, int spacedim>
void nargil::distributed_vector<dim, spacedim>::reinit_global_vec(
  LA::MPI::Vector &global_nodal_vec)
{
  dealii::IndexSet active_idx_set;
  dealii::DoFTools::extract_locally_relevant_dofs(*my_dof_handler,
                                                  active_idx_set);
  global_nodal_vec.reinit(idx_set, active_idx_set, *my_comm);
}

//
//

template <int dim, int spacedim>
void nargil::distributed_vector<dim, spacedim>::copy_to_global_vec(
  LA::MPI::Vector &global_nodal_vec, const bool &do_reinit_global_vec)
{
  if (do_reinit_global_vec)
    reinit_global_vec(global_nodal_vec);
  local_nodal_vec.compress(dealii::VectorOperation::insert);
  global_nodal_vec = local_nodal_vec;
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::dealiiTriCell<dim, spacedim>::dealiiTriCell()
{
}

//
//

template <int dim, int spacedim>
nargil::dealiiTriCell<dim, spacedim>
nargil::dealiiTriCell<dim, spacedim>::generic_neighbor(const unsigned i_face)
{
  baseTriCell<dim, spacedim> dealii_nb_it =
    (*this)->has_periodic_neighbor(i_face) ? (*this)->periodic_neighbor(i_face)
                                           : (*this)->neighbor(i_face);
  dealiiTriCell *nb_it = static_cast<dealiiTriCell *>(&dealii_nb_it);
  return (*nb_it);
}

//
//

template <int dim, int spacedim>
bool nargil::dealiiTriCell<dim, spacedim>::generic_neighbor_is_coarser(
  const unsigned i_face)
{
  bool result = (*this)->has_periodic_neighbor(i_face)
                  ? (*this)->periodic_neighbor_is_coarser(i_face)
                  : (*this)->neighbor_is_coarser(i_face);
  return result;
}

//
//

template <int dim, int spacedim>
unsigned nargil::dealiiTriCell<dim, spacedim>::generic_neighbor_face_no(
  const unsigned i_face)
{
  unsigned face_no = (*this)->has_periodic_neighbor(i_face)
                       ? (*this)->periodic_neighbor_face_no(i_face)
                       : (*this)->neighbor_face_no(i_face);
  return face_no;
}

//
//

template <int dim, int spacedim>
nargil::dealiiTriCell<dim, spacedim>
nargil::dealiiTriCell<dim, spacedim>::generic_neighbor_child_on_subface(
  const unsigned i_face, const unsigned i_subface)
{
  baseTriCell<dim, spacedim> dealii_nb_it =
    (*this)->has_periodic_neighbor(i_face)
      ? (*this)->periodic_neighbor_child_on_subface(i_face, i_subface)
      : (*this)->neighbor_child_on_subface(i_face, i_subface);
  dealiiTriCell *nb_it = static_cast<dealiiTriCell *>(&dealii_nb_it);
  return (*nb_it);
}
