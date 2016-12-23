#include "cell_class.hpp"

template <int dim, int spacedim>
Cell<dim, spacedim>::Cell(dealiiCell &inp_cell,
                          const unsigned &id_num_,
                          const unsigned &poly_order_)
  : n_faces(dealii::GeometryInfo<dim>::faces_per_cell),
    poly_order(poly_order_),
    n_face_bases(pow(poly_order + 1, dim - 1)),
    n_cell_bases(pow(poly_order + 1, dim)),
    id_num(id_num_),
    dof_names_on_faces(n_faces),
    half_range_flag(n_faces, 0),
    face_owner_rank(n_faces, -1),
    dealii_cell(inp_cell),
    dofs_ID_in_this_rank(n_faces),
    dofs_ID_in_all_ranks(n_faces),
    BCs(n_faces, BC::not_set)
{
  std::stringstream ss_id;
  ss_id << inp_cell->id();
  cell_id = ss_id.str();
}

template <int dim, int spacedim>
Cell<dim, spacedim>::Cell(Cell &&inp_cell) noexcept
  : n_faces(std::move(inp_cell.n_faces)),
    poly_order(std::move(inp_cell.poly_order)),
    n_face_bases(std::move(inp_cell.n_face_bases)),
    n_cell_bases(std::move(inp_cell.n_cell_bases)),
    id_num(std::move(inp_cell.id_num)),
    dof_names_on_faces(std::move(inp_cell.dof_names_on_faces)),
    cell_id(std::move(inp_cell.cell_id)),
    half_range_flag(std::move(inp_cell.half_range_flag)),
    face_owner_rank(std::move(inp_cell.face_owner_rank)),
    dealii_cell(std::move(inp_cell.dealii_cell)),
    dofs_ID_in_this_rank(std::move(inp_cell.dofs_ID_in_this_rank)),
    dofs_ID_in_all_ranks(std::move(inp_cell.dofs_ID_in_all_ranks)),
    BCs(std::move(inp_cell.BCs))
{
}

template <int dim, int spacedim>
Cell<dim, spacedim>::~Cell()
{
}

/*
template <int dim, int spacedim>
template <template <int> class type_of_cell>
std::unique_ptr<Cell<dim, spacedim> >
Cell<dim, spacedim>::make_cell(dealiiCell &inp_cell,
                               const unsigned &id_num_,
                               const unsigned &poly_order_,
                               hdg_model<dim, type_of_cell> *model_)
{
  return std::unique_ptr<type_of_cell<dim> >(
    new type_of_cell<dim>(inp_cell, id_num_, poly_order_, model_));
}

template <int dim, int spacedim>
template <template <int> class type_of_cell>
std::unique_ptr<Cell<dim, spacedim> >
Cell<dim, spacedim>::make_cell(dealiiCell &inp_cell,
                               const unsigned &id_num_,
                               const unsigned &poly_order_,
                               explicit_hdg_model<dim, type_of_cell> *model_)
{
  return std::unique_ptr<type_of_cell<dim> >(
    new type_of_cell<dim>(inp_cell, id_num_, poly_order_, model_));
}

template <int dim, int spacedim>
template <template <int> class type_of_cell>
std::unique_ptr<Cell<dim, spacedim> > Cell<dim, spacedim>::make_cell(
  dealiiCell &inp_cell,
  const unsigned &id_num_,
  const unsigned &poly_order_,
  hdg_model_with_explicit_rk<dim, type_of_cell> *model_)
{
  return std::unique_ptr<type_of_cell<dim> >(
    new type_of_cell<dim>(inp_cell, id_num_, poly_order_, model_));
}
*/

/*
template <int dim, int spacedim>
void Cell<dim, spacedim>::attach_FEValues(FE_val_ptr &cell_quad_fe_vals_,
                                          FEFace_val_ptr &face_quad_fe_vals_,
                                          FE_val_ptr &cell_supp_fe_vals_,
                                          FEFace_val_ptr &face_supp_fe_vals_)
{
  cell_quad_fe_vals = std::move(cell_quad_fe_vals_);
  face_quad_fe_vals = std::move(face_quad_fe_vals_);
  cell_supp_fe_vals = std::move(cell_supp_fe_vals_);
  face_supp_fe_vals = std::move(face_supp_fe_vals_);
}

template <int dim, int spacedim>
void Cell<dim, spacedim>::detach_FEValues(FE_val_ptr &cell_quad_fe_vals_,
                                          FEFace_val_ptr &face_quad_fe_vals_,
                                          FE_val_ptr &cell_supp_fe_vals_,
                                          FEFace_val_ptr &face_supp_fe_vals_)
{
  cell_quad_fe_vals_ = std::move(cell_quad_fe_vals);
  face_quad_fe_vals_ = std::move(face_quad_fe_vals);
  cell_supp_fe_vals_ = std::move(cell_supp_fe_vals);
  face_supp_fe_vals_ = std::move(face_supp_fe_vals);
}
*/

/*
template <int dim, int spacedim>
double
Cell<dim, spacedim>::get_error_in_cell(const TimeFunction<dim, double> &func,
                                       const Eigen::MatrixXd &input_vector,
                                       const double &time)
{
  double error = 0;
  const std::vector<dealii::Point<dim> > &points_loc =
    cell_quad_fe_vals->get_quadrature_points();
  const std::vector<double> &JxWs = cell_quad_fe_vals->get_JxW_values();
  assert(points_loc.size() == JxWs.size());
  assert(input_vector.rows() == the_elem_basis->n_polys);
  //  assert((long int)points_loc.size() == mode_to_Qpoint_matrix.rows());
  Eigen::MatrixXd values_at_Nodes =
    the_elem_basis->get_dof_vals_at_quads(input_vector);
  for (unsigned i_point = 0; i_point < JxWs.size(); ++i_point)
  {
    error += (func.value(points_loc[i_point], points_loc[i_point], time) -
              values_at_Nodes(i_point, 0)) *
             (func.value(points_loc[i_point], points_loc[i_point], time) -
              values_at_Nodes(i_point, 0)) *
             JxWs[i_point];
  }
  return error;
}

template <int dim, int spacedim>
template <int func_out_dim>
double Cell<dim, spacedim>::get_error_in_cell(
  const TimeFunction<dim, dealii::Tensor<1, func_out_dim> > &func,
  const Eigen::MatrixXd &modal_vector,
  const double &time)
{
  double error = 0;
  const std::vector<dealii::Point<dim> > &points_loc =
    cell_quad_fe_vals->get_quadrature_points();
  const std::vector<double> &JxWs = cell_quad_fe_vals->get_JxW_values();
  unsigned n_unknowns = the_elem_basis->n_polys;
  assert(points_loc.size() == JxWs.size());
  assert(modal_vector.rows() == func_out_dim * n_unknowns);
  //  assert((long int)points_loc.size() == mode_to_Qpoint_matrix.rows());
  std::vector<Eigen::MatrixXd> values_at_Nodes(func_out_dim);
  for (unsigned i_dim = 0; i_dim < func_out_dim; ++i_dim)
    values_at_Nodes[i_dim] = the_elem_basis->get_dof_vals_at_quads(
      modal_vector.block(i_dim * n_unknowns, 0, n_unknowns, 1));
  for (unsigned i_point = 0; i_point < JxWs.size(); ++i_point)
  {
    dealii::Tensor<1, func_out_dim> temp_val;
    for (unsigned i_dim = 0; i_dim < func_out_dim; ++i_dim)
    {
      temp_val[i_dim] = values_at_Nodes[i_dim](i_point, 0);
    }
    dealii::Tensor<1, func_out_dim> func_val =
      func.value(points_loc[i_point], points_loc[i_point], time);
    error += (func_val - temp_val) * (func_val - temp_val) * JxWs[i_point];
  }
  return error;
}

template <int dim, int spacedim>
double Cell<dim, spacedim>::get_error_on_faces(
  const TimeFunction<dim, double> &, const Eigen::MatrixXd &, const double &)
{
  assert(false);
}

template <int dim, int spacedim>
template <int func_out_dim>
double Cell<dim, spacedim>::get_error_on_faces(
  const TimeFunction<dim, dealii::Tensor<1, func_out_dim> > &func,
  const Eigen::MatrixXd &modal_vector,
  const double &time)
{
  double error = 0;
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    this->reinit_face_fe_vals(i_face);
    const std::vector<dealii::Point<dim> > &points_loc =
      face_quad_fe_vals->get_quadrature_points();
    const std::vector<double> &JxWs = face_quad_fe_vals->get_JxW_values();
    unsigned n_unknowns = the_face_basis->n_polys;
    assert(points_loc.size() == JxWs.size());
    assert(modal_vector.rows() == func_out_dim * n_unknowns * n_faces);
    //    assert((long int)points_loc.size() == mode_to_Qpoint_matrix.rows());
    std::vector<Eigen::MatrixXd> values_at_Nodes(func_out_dim);
    for (unsigned i_dim = 0; i_dim < func_out_dim; ++i_dim)
    {
      unsigned i_row = (i_face * func_out_dim + i_dim) * n_face_bases;
      values_at_Nodes[i_dim] = the_face_basis->get_dof_vals_at_quads(
        modal_vector.block(i_row, 0, n_unknowns, 1));
    }
    for (unsigned i_point = 0; i_point < JxWs.size(); ++i_point)
    {
      dealii::Tensor<1, func_out_dim> temp_val;
      for (unsigned i_dim = 0; i_dim < func_out_dim; ++i_dim)
      {
        temp_val[i_dim] = values_at_Nodes[i_dim](i_point, 0);
      }
      dealii::Tensor<1, func_out_dim> func_val =
        func.value(points_loc[i_point], points_loc[i_point], time);
      error += (func_val - temp_val) * (func_val - temp_val) * JxWs[i_point];
    }
  }
  return error;
}

template <int dim, int spacedim>
void Cell<dim, spacedim>::internal_vars_errors(const eigen3mat &,
                                               const eigen3mat &,
                                               double &,
                                               double &)
{
}
*/

template <int dim, int spacedim>
void Cell<dim, spacedim>::reinit_cell_fe_vals()
{
  cell_quad_fe_vals->reinit(dealii_cell);
  cell_supp_fe_vals->reinit(dealii_cell);
}

template <int dim, int spacedim>
void Cell<dim, spacedim>::reinit_face_fe_vals(unsigned i_face)
{
  face_quad_fe_vals->reinit(dealii_cell, i_face);
  face_supp_fe_vals->reinit(dealii_cell, i_face);
}

/*
template <int dim, int spacedim>
template <typename BasisType, typename func_out_type>
void Cell<dim, spacedim>::project_essential_BC_to_face(
  const TimeFunction<dim, func_out_type> &func,
  const poly_space_basis<BasisType, dim - 1> &the_basis,
  const std::vector<double> &weights,
  mtl::vec::dense_vector<func_out_type> &vec,
  const double &time)
{
  if (std::is_same<JacobiPolys<dim - 1>, BasisType>::value)
  {
    std::vector<dealii::Point<dim> > integration_points_loc =
      this->face_quad_fe_vals->get_quadrature_points();
    assert(the_basis.n_quads == integration_points_loc.size());
    assert(integration_points_loc.size() == weights.size());
    vec.change_dim(the_basis.n_polys);
    vec = 0;
    for (unsigned i1 = 0; i1 < weights.size(); ++i1)
    {
      const eigen3mat &Nj0 = the_basis.get_func_vals_at_iquad(i1);
      mtl::vec::dense_vector<double> Nj(the_basis.n_polys,
                                        (double *)Nj0.data());
      const func_out_type &value = func.value(
        integration_points_loc[i1], integration_points_loc[i1], time);
      vec += weights[i1] * value * Nj;
    }
  }
  else if (std::is_same<LagrangePolys<dim - 1>, BasisType>::value)
  {
    std::vector<dealii::Point<dim> > support_points_loc =
      this->face_supp_fe_vals->get_quadrature_points();
    assert(support_points_loc.size() == the_basis.n_polys);
    vec.change_dim(the_basis.n_polys);
    unsigned counter = 0;
    for (auto &&support_point : support_points_loc)
      vec[counter++] = func.value(support_point, support_point, time);
  }
}

template <int dim, int spacedim>
template <typename BasisType, typename func_out_type>
void Cell<dim, spacedim>::project_func_to_face(
  const TimeFunction<dim, func_out_type> &func,
  const poly_space_basis<BasisType, dim - 1> &the_basis,
  const std::vector<double> &weights,
  mtl::vec::dense_vector<func_out_type> &vec,
  const unsigned &i_face,
  const double &time)
{
  if (std::is_same<JacobiPolys<dim - 1>, BasisType>::value)
  {
    std::vector<dealii::Point<dim> > integration_points_loc =
      this->face_quad_fe_vals->get_quadrature_points();
    std::vector<dealii::Point<dim - 1> > face_quad_points =
      this->face_quad_bundle->get_points();
    assert(the_basis.n_quads == integration_points_loc.size());
    assert(integration_points_loc.size() == weights.size());
    vec.change_dim(the_basis.n_polys);
    vec = 0;
    for (unsigned i1 = 0; i1 < weights.size(); ++i1)
    {
      const std::vector<double> &face_basis_at_iquad =
        this->the_face_basis->value(face_quad_points[i1],
                                    this->half_range_flag[i_face]);
      mtl::vec::dense_vector<double> Nj(the_basis.n_polys,
                                        (double *)face_basis_at_iquad.data());
      const func_out_type &value = func.value(
        integration_points_loc[i1], integration_points_loc[i1], time);
      vec += weights[i1] * value * Nj;
    }
  }
  else if (std::is_same<LagrangePolys<dim - 1>, BasisType>::value)
  {
    std::vector<dealii::Point<dim> > support_points_loc =
      this->face_supp_fe_vals->get_quadrature_points();
    assert(support_points_loc.size() == the_basis.n_polys);
    vec.change_dim(the_basis.n_polys);
    unsigned counter = 0;
    for (auto &&support_point : support_points_loc)
      vec[counter++] = func.value(support_point, support_point, time);
  }
}

template <int dim, int spacedim>
template <typename BasisType, typename func_out_type>
void Cell<dim, spacedim>::project_flux_BC_to_face(
  const Function<dim, func_out_type> &func,
  const poly_space_basis<BasisType, dim - 1> &the_basis,
  const std::vector<double> &weights,
  mtl::vec::dense_vector<func_out_type> &vec)
{
  if (std::is_same<JacobiPolys<dim - 1>, BasisType>::value)
  {
    std::vector<dealii::Point<dim> > integration_points_loc =
      this->face_quad_fe_vals->get_quadrature_points();
    std::vector<dealii::Point<dim> > normals_at_integration =
      this->face_quad_fe_vals->get_normal_vectors();
    assert(the_basis.n_quads == integration_points_loc.size());
    assert(integration_points_loc.size() == weights.size());
    vec.change_dim(the_basis.n_polys);
    vec = 0;
    for (unsigned i1 = 0; i1 < weights.size(); ++i1)
    {
      const eigen3mat &Nj0 = the_basis.get_func_vals_at_iquad(i1);
      mtl::vec::dense_vector<double> Nj(the_basis.n_polys,
                                        (double *)Nj0.data());
      const func_out_type &value =
        func.value(integration_points_loc[i1], normals_at_integration[i1]);
      vec += weights[i1] * value * Nj;
    }
  }
  else if (std::is_same<LagrangePolys<dim - 1>, BasisType>::value)
  {
    std::vector<dealii::Point<dim> > support_points_loc =
      this->face_supp_fe_vals->get_quadrature_points();
    std::vector<dealii::Point<dim> > normals_at_supports =
      this->face_supp_fe_vals->get_normal_vectors();
    assert(support_points_loc.size() == the_basis.n_polys);
    vec.change_dim(the_basis.n_polys);
    vec = 0;
    unsigned counter = 0;
    for (auto &&support_point : support_points_loc)
    {
      vec[counter] = func.value(support_point, normals_at_supports[counter]);
      counter++;
    }
  }
}

template <int dim, int spacedim>
template <typename BasisType, typename func_out_type>
void Cell<dim, spacedim>::project_to_elem_basis(
  const TimeFunction<dim, func_out_type> &func,
  const poly_space_basis<BasisType, dim> &the_basis,
  const std::vector<double> &weights,
  mtl::vec::dense_vector<func_out_type> &vec,
  const double &time)
{
  if (std::is_same<JacobiPolys<dim>, BasisType>::value)
  {
    std::vector<dealii::Point<dim> > integration_points_loc =
      this->cell_quad_fe_vals->get_quadrature_points();
    assert(the_basis.n_quads == integration_points_loc.size());
    assert(integration_points_loc.size() == weights.size());
    vec.change_dim(the_basis.n_polys);
    vec = 0;
    for (unsigned i1 = 0; i1 < weights.size(); ++i1)
    {
      const eigen3mat &Nj0 = the_basis.get_func_vals_at_iquad(i1);
      mtl::vec::dense_vector<double> Nj(the_basis.n_polys,
                                        (double *)Nj0.data());
      const func_out_type &value = func.value(
        integration_points_loc[i1], integration_points_loc[i1], time);
      vec += (weights[i1] * value) * Nj;
    }
  }
  else if (std::is_same<LagrangePolys<dim>, BasisType>::value)
  {
    std::vector<dealii::Point<dim> > support_points_loc =
      this->cell_supp_fe_vals->get_quadrature_points();
    assert(support_points_loc.size() == the_basis.n_polys);
    vec.change_dim(the_basis.n_polys);
    unsigned counter = 0;
    for (auto &&support_point : support_points_loc)
      vec[counter++] = func.value(support_point, support_point, time);
  }
}
*/

template <int dim, int spacedim>
void Cell<dim, spacedim>::assign_local_global_cell_data(
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

template <int dim, int spacedim>
void Cell<dim, spacedim>::assign_ghost_cell_data(const unsigned &i_face,
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

template <int dim, int spacedim>
void Cell<dim, spacedim>::assign_local_cell_data(const unsigned &i_face,
                                                 const unsigned &local_num_,
                                                 const int &comm_rank_,
                                                 const unsigned &half_range_)
{
  face_owner_rank[i_face] = comm_rank_;
  half_range_flag[i_face] = half_range_;
  for (unsigned i_dof = 0; i_dof < dof_names_on_faces[i_face].count(); ++i_dof)
    dofs_ID_in_this_rank[i_face].push_back(local_num_ + i_dof);
}
