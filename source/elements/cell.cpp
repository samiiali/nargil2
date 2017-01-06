#include "../../include/elements/cell.hpp"

template <int dim, int spacedim>
nargil::cell_worker<dim, spacedim>::cell_worker(
  nargil::cell<dim, spacedim> *const in_cell)
  : my_cell(in_cell)
{
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::base_hdg_worker<dim, spacedim>::base_hdg_worker(
  cell<dim, spacedim> *in_cell)
  : cell_worker<dim, spacedim>(in_cell),
    dofs_ID_in_this_rank(2 * dim),
    dofs_ID_in_all_ranks(2 * dim),
    BCs(2 * dim, boundary_condition::not_set),
    dof_names_on_faces(2 * dim),
    half_range_flag(2 * dim, 0),
    face_owner_rank(2 * dim, -1)
{
}

//
//

template <int dim, int spacedim>
nargil::base_hdg_worker<dim, spacedim>::~base_hdg_worker()
{
}

//
//

template <int dim, int spacedim>
void nargil::base_hdg_worker<dim, spacedim>::assign_local_global_cell_data(
  const unsigned &i_face,
  const unsigned &local_num_,
  const unsigned &global_num_,
  const unsigned &comm_rank_,
  const unsigned &half_range_)
{
  face_owner_rank[i_face] = comm_rank_;
  half_range_flag[i_face] = half_range_;
  for (unsigned i_dof = 0; i_dof < dof_names_on_faces[i_face].count(); ++i_dof)
  {
    dofs_ID_in_this_rank[i_face].push_back(local_num_ + i_dof);
    dofs_ID_in_all_ranks[i_face].push_back(global_num_ + i_dof);
  }
}

//
//

template <int dim, int spacedim>
void nargil::base_hdg_worker<dim, spacedim>::assign_ghost_cell_data(
  const unsigned &i_face,
  const int &local_num_,
  const int &global_num_,
  const unsigned &comm_rank_,
  const unsigned &half_range_)
{
  face_owner_rank[i_face] = comm_rank_;
  half_range_flag[i_face] = half_range_;
  for (unsigned i_dof = 0; i_dof < dof_names_on_faces[i_face].count(); ++i_dof)
  {
    dofs_ID_in_this_rank[i_face].push_back(local_num_ - i_dof);
    dofs_ID_in_all_ranks[i_face].push_back(global_num_ - i_dof);
  }
}

//
//

template <int dim, int spacedim>
void nargil::base_hdg_worker<dim, spacedim>::assign_local_cell_data(
  const unsigned &i_face,
  const unsigned &local_num_,
  const int &comm_rank_,
  const unsigned &half_range_)
{
  face_owner_rank[i_face] = comm_rank_;
  half_range_flag[i_face] = half_range_;
  for (unsigned i_dof = 0; i_dof < dof_names_on_faces[i_face].count(); ++i_dof)
    dofs_ID_in_this_rank[i_face].push_back(local_num_ + i_dof);
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::cell<dim, spacedim>::cell(dealii_cell_type &inp_cell,
                                  const unsigned id_num_,
                                  base_model *model_)
  : n_faces(dealii::GeometryInfo<dim>::faces_per_cell),
    id_num(id_num_),
    dealii_cell(inp_cell),
    my_model(model_)
{
  std::stringstream ss_id;
  ss_id << inp_cell->id();
  cell_id = ss_id.str();
}

//
//

template <int dim, int spacedim>
nargil::cell<dim, spacedim>::cell(cell &&inp_cell) noexcept
  : n_faces(std::move(inp_cell.n_faces)),
    id_num(std::move(inp_cell.id_num)),
    cell_id(std::move(inp_cell.cell_id)),
    dealii_cell(std::move(inp_cell.dealii_cell)),
    my_model(inp_cell.my_model)
{
}

//
//

template <int dim, int spacedim> nargil::cell<dim, spacedim>::~cell() {}

//
//

template <int dim, int spacedim>
template <typename ModelEq, typename BasisType>
std::unique_ptr<ModelEq>
nargil::cell<dim, spacedim>::create(dealii_cell_type &inp_cell,
                                    const unsigned id_num_,
                                    BasisType *basis,
                                    base_model *model_)
{
  return std::unique_ptr<ModelEq>(
    new ModelEq(inp_cell, id_num_, basis, model_, BasisType::get_options()));
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
