#include "../../include/elements/cell.hpp"

template <int dim, int spacedim>
nargil::cell_manager<dim, spacedim>::cell_manager(
  const cell<dim, spacedim> *in_cell)
  : my_cell(in_cell)
{
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::hybridized_cell_manager<dim, spacedim>::hybridized_cell_manager(
  const cell<dim, spacedim> *in_cell)
  : cell_manager<dim, spacedim>(in_cell),
    unkns_id_in_this_rank(2 * dim),
    unkns_id_in_all_ranks(2 * dim),
    BCs(2 * dim, boundary_condition::not_set),
    dof_status_on_faces(2 * dim),
    half_range_flag(2 * dim, 0),
    face_owner_rank(2 * dim, -1)
{
}

//
//

template <int dim, int spacedim>
nargil::hybridized_cell_manager<dim, spacedim>::~hybridized_cell_manager()
{
}

//
//

template <int dim, int spacedim>
void nargil::hybridized_cell_manager<dim, spacedim>::set_cell_properties(
  const unsigned i_face, const unsigned in_comm_rank,
  const unsigned in_half_range)
{
  face_owner_rank[i_face] = in_comm_rank;
  half_range_flag[i_face] = in_half_range;
  face_visited[i_face] = 1;
}

//
//

template <int dim, int spacedim>
void nargil::hybridized_cell_manager<dim, spacedim>::set_owned_unkn_ids(
  const unsigned i_face,
  const unsigned local_num_,
  const unsigned global_num_,
  const std::vector<unsigned> &n_unkns_per_dofs)
{
  unsigned n_unkns = 0;
  for (unsigned i_dof = 0; i_dof < dof_status_on_faces[i_face].size(); ++i_dof)
  {
    if (dof_status_on_faces[i_face][i_dof])
    {
      for (unsigned i_unkn = 0; i_unkn < n_unkns_per_dofs[i_dof]; ++i_unkn)
      {
        unkns_id_in_this_rank[i_face].push_back(local_num_ + n_unkns + i_unkn);
        unkns_id_in_all_ranks[i_face].push_back(global_num_ + n_unkns + i_unkn);
      }
      n_unkns += n_unkns_per_dofs[i_dof];
    }
    else
    {
      for (unsigned i_unkn = 0; i_unkn < n_unkns_per_dofs[i_dof]; ++i_unkn)
      {
        unkns_id_in_this_rank[i_face].push_back(-1);
        unkns_id_in_all_ranks[i_face].push_back(-1);
      }
    }
  }
}

//
//

template <int dim, int spacedim>
void nargil::hybridized_cell_manager<dim, spacedim>::set_local_unkn_ids(
  const unsigned i_face,
  const unsigned local_num_,
  const std::vector<unsigned> &n_unkns_per_dofs)
{
  unsigned n_unkns = 0;
  for (unsigned i_dof = 0; i_dof < dof_status_on_faces[i_face].size(); ++i_dof)
  {
    if (dof_status_on_faces[i_face][i_dof])
    {
      for (unsigned i_unkn = 0; i_unkn < n_unkns_per_dofs[i_dof]; ++i_unkn)
      {
        unkns_id_in_this_rank[i_face].push_back(local_num_ + n_unkns + i_unkn);
      }
      n_unkns += n_unkns_per_dofs[i_dof];
    }
    else
    {
      for (unsigned i_unkn = 0; i_unkn < n_unkns_per_dofs[i_dof]; ++i_unkn)
      {
        unkns_id_in_this_rank[i_face].push_back(-1);
      }
    }
  }
}

//
//

template <int dim, int spacedim>
void nargil::hybridized_cell_manager<dim, spacedim>::set_ghost_unkn_ids(
  const unsigned i_face, const int ghost_num_,
  const std::vector<unsigned> &n_unkns_per_dofs)
{
  unsigned n_unkns = 0;
  for (unsigned i_dof = 0; i_dof < dof_status_on_faces[i_face].size(); ++i_dof)
  {
    if (dof_status_on_faces[i_face][i_dof])
    {
      for (unsigned i_unkn = 0; i_unkn < n_unkns_per_dofs[i_dof]; ++i_unkn)
      {
        unkns_id_in_this_rank[i_face].push_back(ghost_num_ - n_unkns - i_unkn);
        unkns_id_in_all_ranks[i_face].push_back(ghost_num_ - n_unkns - i_unkn);
      }
      n_unkns += n_unkns_per_dofs[i_dof];
    }
    else
    {
      for (unsigned i_unkn = 0; i_unkn < n_unkns_per_dofs[i_dof]; ++i_unkn)
      {
        unkns_id_in_this_rank[i_face].push_back(-1);
        unkns_id_in_all_ranks[i_face].push_back(-1);
      }
    }
  }
}

//
//

template <int dim, int spacedim>
void nargil::hybridized_cell_manager<dim, spacedim>::set_nonlocal_unkn_ids(
  const unsigned i_face,
  const int global_num_,
  const std::vector<unsigned> &n_unkns_per_dofs)
{
  assert(global_num_ >= 0);
  unsigned n_unkns = 0;
  for (unsigned i_dof = 0; i_dof < dof_status_on_faces[i_face].size(); ++i_dof)
  {
    if (dof_status_on_faces[i_face][i_dof])
    {
      for (unsigned i_unkn = 0; i_unkn < n_unkns_per_dofs[i_dof]; ++i_unkn)
      {
        unkns_id_in_all_ranks[i_face].push_back(global_num_ + n_unkns + i_unkn);
      }
      n_unkns += n_unkns_per_dofs[i_dof];
    }
    else
    {
      for (unsigned i_unkn = 0; i_unkn < n_unkns_per_dofs[i_dof]; ++i_unkn)
      {
        unkns_id_in_all_ranks[i_face].push_back(-1);
      }
    }
  }
}

//
//

template <int dim, int spacedim>
void nargil::hybridized_cell_manager<dim, spacedim>::offset_global_unkn_ids(
  const int dofs_count_be4_rank)
{
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    for (unsigned i_unkn = 0; i_unkn < unkns_id_in_all_ranks[i_face].size();
         ++i_unkn)
      if (unkns_id_in_all_ranks[i_face][i_unkn] >= 0)
        unkns_id_in_all_ranks[i_face][i_unkn] += dofs_count_be4_rank;
}

//
//

template <int dim, int spacedim>
bool nargil::hybridized_cell_manager<dim, spacedim>::face_is_not_visited(
  const unsigned i_face)
{
  return !(face_visited[i_face]);
}

//
//

template <int dim, int spacedim>
unsigned
nargil::hybridized_cell_manager<dim, spacedim>::get_n_open_unknowns_on_face(
  const unsigned i_face, const std::vector<unsigned> &n_unkns_per_dofs)
{
  unsigned n_unkns = 0;
  for (unsigned i_dof = 0; i_dof < dof_status_on_faces[i_face].size(); ++i_dof)
  {
    unsigned dof_is_open = (unsigned)dof_status_on_faces[i_face][i_dof];
    n_unkns += dof_is_open * n_unkns_per_dofs[i_dof];
  }
  return n_unkns;
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::cell<dim, spacedim>::cell(dealiiCell &inp_cell,
                                  const unsigned id_num_,
                                  const base_model *model_)
  : n_faces(2 * dim), id_num(id_num_), dealii_cell(inp_cell), my_model(model_)
{
  std::stringstream ss_id;
  ss_id << inp_cell->id();
  cell_id = ss_id.str();
}

//
//

template <int dim, int spacedim> nargil::cell<dim, spacedim>::~cell() {}

//
//

template <int dim, int spacedim>
template <typename ModelEq, typename BasisType>
std::unique_ptr<ModelEq>
nargil::cell<dim, spacedim>::create(dealiiCell &in_cell, const unsigned id_num_,
                                    const BasisType &basis,
                                    base_model *in_model)
{
  std::unique_ptr<ModelEq> the_cell(
    new ModelEq(in_cell, id_num_, &basis, in_model));
  the_cell->template init_manager<typename BasisType::CellManagerType>();
  return std::move(the_cell);
}

//
//

template <int dim, int spacedim>
void nargil::cell<dim, spacedim>::reinit_cell_fe_vals()
{
  // cell_quad_fe_vals->reinit(dealii_cell);
  // cell_supp_fe_vals->reinit(dealii_cell);
}

//
//

template <int dim, int spacedim>
void nargil::cell<dim, spacedim>::reinit_face_fe_vals(unsigned)
{
  // face_quad_fe_vals->reinit(dealii_cell, i_face);
  // face_supp_fe_vals->reinit(dealii_cell, i_face);
}

//
//
