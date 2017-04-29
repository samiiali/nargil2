#include "../../include/elements/reactive_interface.hpp"

//
//

template <int dim, int spacedim>
nargil::reactive_interface<dim, spacedim>::reactive_interface(
  dealiiTriCell<dim, spacedim> *inp_cell,
  const unsigned in_id_num,
  base_basis<dim, spacedim> *in_basis)
  : cell<dim, spacedim>(inp_cell, in_id_num), my_basis(in_basis)
{
}

//
//

template <int dim, int spacedim>
nargil::reactive_interface<dim, spacedim>::~reactive_interface()
{
}

//
//

template <int dim, int spacedim>
template <typename CellManagerType, typename BasisType>
void nargil::reactive_interface<
  dim, spacedim>::reactive_interface::init_manager(const BasisType *in_basis)
{
  my_manager = std::move(
    std::unique_ptr<CellManagerType>(new CellManagerType(this, in_basis)));
}

//
//

template <int dim, int spacedim>
void nargil::reactive_interface<dim, spacedim>::assign_data(
  reactive_interface *in_cell, data *in_data)
{
  in_cell->my_data = in_data;
}

//
//

template <int dim, int spacedim>
template <typename CellManagerType>
CellManagerType *nargil::reactive_interface<dim, spacedim>::get_manager()
{
  return static_cast<CellManagerType *>(my_manager.get());
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
const BasisType *nargil::reactive_interface<dim, spacedim>::get_basis() const
{
  return static_cast<const BasisType *>(my_basis);
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::reactive_interface<dim, spacedim>::viz_data::viz_data(
  const MPI_Comm in_comm,
  const dealii::DoFHandler<dim, spacedim> *in_dof_handler,
  const dealii::LinearAlgebraPETSc::MPI::Vector *in_viz_sol,
  const std::string &in_filename, const std::vector<std::string> &in_var_names)
  : my_comm(in_comm),
    my_dof_handler(in_dof_handler),
    my_viz_sol(in_viz_sol),
    my_out_filename(in_filename),
    my_var_names(in_var_names)
{
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::reactive_interface<dim, spacedim>::hdg_polybasis::hdg_polybasis(
  const unsigned poly_order, const unsigned quad_order)
  : base_basis<dim, spacedim>(),
    local_fe(dealii::FE_DGQ<dim>(poly_order), 1,
             dealii::FE_DGQ<dim>(poly_order), dim),
    trace_fe(poly_order),
    refn_fe(poly_order),
    viz_fe(std::vector<const dealii::FiniteElement<dim> *>(8, &refn_fe),
           std::vector<unsigned>{1, dim, 1, dim, 1, dim, 1, dim}),
    trace_fe_face_val(pow(2, (dim - 1)) + 1),
    local_fe_val_on_faces(2 * dim)
{
  dealii::QGauss<dim> cell_quad(quad_order);
  dealii::QGauss<dim - 1> face_quad(quad_order);
  //
  dealii::UpdateFlags fe_val_flags(
    dealii::update_values | dealii::update_gradients |
    dealii::update_JxW_values | dealii::update_quadrature_points);
  //
  dealii::UpdateFlags fe_val_flags_at_supp(dealii::update_values |
                                           dealii::update_quadrature_points);
  //
  dealii::UpdateFlags fe_face_val_flags(
    dealii::update_values | dealii::update_JxW_values |
    dealii::update_normal_vectors | dealii::update_quadrature_points);
  //
  std::unique_ptr<dealii::FEValues<dim> > local_fe_val_temp(
    new dealii::FEValues<dim>(local_fe, cell_quad, fe_val_flags));
  //
  std::unique_ptr<dealii::FEValues<dim> > local_fe_val_at_supp_temp(
    new dealii::FEValues<dim>(local_fe, local_fe.get_unit_support_points(),
                              fe_val_flags_at_supp));
  //
  dealii::QGaussLobatto<dim - 1> nodal_q(poly_order + 1);
  std::unique_ptr<dealii::FEFaceValues<dim> > trace_fe_face_val_supp_temp(
    new dealii::FEFaceValues<dim>(trace_fe, nodal_q, fe_face_val_flags));
  //
  local_fe_val_in_cell = std::move(local_fe_val_temp);
  local_fe_val_at_cell_supp = std::move(local_fe_val_at_supp_temp);
  trace_fe_face_val_at_supp = std::move(trace_fe_face_val_supp_temp);
  //
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    dealii::Quadrature<dim> projected_i_face_quad =
      dealii::QProjector<dim>::project_to_face(face_quad, i_face);
    std::unique_ptr<dealii::FEValues<dim> > local_fe_vel_on_i_face(
      new dealii::FEValues<dim>(local_fe, projected_i_face_quad, fe_val_flags));
    local_fe_val_on_faces[i_face] = std::move(local_fe_vel_on_i_face);
  }
  //
  for (unsigned i_half = 0; i_half <= pow(2, dim - 1); ++i_half)
  {
    std::vector<dealii::Point<dim - 1> > half_face_q_points(
      face_quad.get_points());
    for (dealii::Point<dim - 1> &p : half_face_q_points)
      p = adjusted_subface_quad_points(p, i_half);
    dealii::Quadrature<dim - 1> half_face_quad(half_face_q_points,
                                               face_quad.get_weights());
    std::unique_ptr<dealii::FEFaceValues<dim> > half_trace_fe_face_val_temp(
      new dealii::FEFaceValues<dim>(trace_fe, half_face_quad,
                                    fe_face_val_flags));
    trace_fe_face_val[i_half] = std::move(half_trace_fe_face_val_temp);
  }
}

//
//

template <int dim, int spacedim>
nargil::reactive_interface<dim, spacedim>::hdg_polybasis::~hdg_polybasis()
{
}

//
//

template <int dim, int spacedim>
dealii::Point<dim - 1>
nargil::reactive_interface<dim, spacedim>::hdg_polybasis::
  adjusted_subface_quad_points(const dealii::Point<dim - 1> &in_point,
                               const unsigned half_range)
{
  assert(half_range <= pow(2, in_point.dimension));
  dealii::Point<dim - 1> out_point(in_point);
  if (half_range != 0)
  {
    if (dim == 2)
    {
      if (half_range == 1)
        out_point(0) = in_point(0) / 2.0;
      if (half_range == 2)
        out_point(0) = 0.5 + in_point(0) / 2.0;
    }
    if (dim == 3)
    {
      if (half_range == 1)
      {
        out_point(0) = in_point(0) / 2.0;
        out_point(1) = in_point(1) / 2.0;
      }
      if (half_range == 2)
      {
        out_point(0) = 0.5 + in_point(0) / 2.0;
        out_point(1) = in_point(1) / 2.0;
      }
      if (half_range == 3)
      {
        out_point(0) = in_point(0) / 2.0;
        out_point(1) = 0.5 + in_point(1) / 2.0;
      }
      if (half_range == 4)
      {
        out_point(0) = 0.5 + in_point(0) / 2.0;
        out_point(1) = 0.5 + in_point(1) / 2.0;
      }
    }
  }
  return out_point;
}

//
//

template <int dim, int spacedim>
unsigned
nargil::reactive_interface<dim, spacedim>::hdg_polybasis::get_n_dofs_per_face()
{
  return 4;
}

//
//

template <int dim, int spacedim>
std::vector<unsigned>
nargil::reactive_interface<dim, spacedim>::hdg_polybasis::get_n_unkns_per_dofs()
  const
{
  // Here, we will have copy elision, DO NOT try to optimize using move
  // semantics.
  std::vector<unsigned> n_unkns_per_dofs(get_n_dofs_per_face());
  for (unsigned &n_unkns : n_unkns_per_dofs)
    n_unkns = trace_fe.dofs_per_face;
  return n_unkns_per_dofs;
}

//
//

template <int dim, int spacedim>
const dealii::FESystem<dim> *
nargil::reactive_interface<dim, spacedim>::hdg_polybasis::get_local_fe() const
{
  return &local_fe;
}

//
//

template <int dim, int spacedim>
const dealii::FE_FaceQ<dim> *
nargil::reactive_interface<dim, spacedim>::hdg_polybasis::get_trace_fe() const
{
  return &trace_fe;
}

//
//

template <int dim, int spacedim>
const dealii::FE_DGQ<dim> *
nargil::reactive_interface<dim, spacedim>::hdg_polybasis::get_refn_fe() const
{
  return &refn_fe;
}

//
//

template <int dim, int spacedim>
const dealii::FESystem<dim> *
nargil::reactive_interface<dim, spacedim>::hdg_polybasis::get_viz_fe() const
{
  return &viz_fe;
}

//
//

template <int dim, int spacedim>
unsigned
nargil::reactive_interface<dim, spacedim>::hdg_polybasis::get_face_quad_size()
  const
{
  return trace_fe_face_val[0]->get_quadrature().size();
}

//
//

template <int dim, int spacedim>
unsigned
nargil::reactive_interface<dim, spacedim>::hdg_polybasis::get_cell_quad_size()
  const
{
  return local_fe_val_in_cell->get_quadrature().size();
}

//
//

template <int dim, int spacedim>
unsigned nargil::reactive_interface<
  dim, spacedim>::hdg_polybasis::n_unkns_per_local_scalar_dof() const
{
  return local_fe.base_element(0).dofs_per_cell;
}

//
//

template <int dim, int spacedim>
unsigned nargil::reactive_interface<
  dim, spacedim>::hdg_polybasis::n_trace_unkns_per_cell_dof() const
{
  return trace_fe.dofs_per_cell;
}

//
//

template <int dim, int spacedim>
unsigned nargil::reactive_interface<
  dim, spacedim>::hdg_polybasis::n_trace_unkns_per_face_dof() const
{
  return trace_fe.dofs_per_face;
}

//
//

template <int dim, int spacedim>
unsigned nargil::reactive_interface<
  dim, spacedim>::hdg_polybasis::n_local_unkns_per_cell() const
{
  return 4 * local_fe.dofs_per_cell;
}

//
//
//
//
//

template <int dim, int spacedim>
template <typename BasisType>
nargil::reactive_interface<dim, spacedim>::hdg_manager<BasisType>::hdg_manager(
  const nargil::reactive_interface<dim, spacedim> *in_cell,
  const BasisType *in_basis)
  : hybridized_cell_manager<dim, spacedim>(in_cell),
    BCs(2 * dim, boundary_condition::not_set),
    my_basis(in_basis)
{
  local_interior_unkn_idx.resize(my_basis->n_local_unkns_per_cell());
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
nargil::reactive_interface<dim,
                           spacedim>::hdg_manager<BasisType>::~hdg_manager()
{
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::assign_my_BCs(BC_Func f)
{
  f(this);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::set_local_interior_unkn_id(unsigned *local_num)
{
  for (unsigned i_unkn = 0; i_unkn < local_interior_unkn_idx.size(); ++i_unkn)
  {
    local_interior_unkn_idx[i_unkn] = *local_num;
    ++(*local_num);
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<BasisType>::
  assemble_my_globals(solvers::base_implicit_solver<dim, spacedim> *in_solver)
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  compute_my_matrices();
  //
  double mu_n = own_cell->my_data->mu_n();
  double c_n =
    own_cell->my_data->alpha_n() * mu_n * own_cell->my_data->lambda_inv2_S();
  /*
  double mu_p = own_cell->my_data->mu_n();
  double c_p = own_cell->my_data->alpha_p() * mu_p *
                   own_cell->my_data->lambda_inv2_S();
  double mu_r = own_cell->my_data->mu_n();
  double c_r = own_cell->my_data->alpha_r() * mu_r *
                   own_cell->my_data->lambda_inv2_E();
  double mu_o = own_cell->my_data->mu_n();
  double c_o = own_cell->my_data->alpha_o() * mu_o *
                   own_cell->my_data->lambda_inv2_E();
  */
  //
  Eigen::MatrixXd A_inv = A1.inverse();
  Eigen::FullPivLU<Eigen::MatrixXd> lu_of_Mat1_n(
    B1.transpose() * (mu_n * A_inv) * B1 + D1 - c_n * D2);
  /*
  Eigen::FullPivLU<Eigen::MatrixXd> lu_of_Mat1_p(
    B1.transpose() * (mu_n * A_inv) * B1 + D1 - c_n * D2);
  Eigen::FullPivLU<Eigen::MatrixXd> lu_of_Mat1_r(
    B1.transpose() * (mu_n * A_inv) * B1 + D1 - c_n * D2);
  Eigen::FullPivLU<Eigen::MatrixXd> lu_of_Mat1_o(
    B1.transpose() * (mu_n * A_inv) * B1 + D1 - c_n * D2);
  */
  Eigen::MatrixXd Mat2 = (B1.transpose() * A_inv * C1 + E1);
  //
  std::vector<int> dof_indices(my_basis->n_trace_unkns_per_cell_dof());
  for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
    for (unsigned i_unkn = 0; i_unkn < my_basis->n_trace_unkns_per_face_dof();
         ++i_unkn)
    {
      int idx1 = i_face * my_basis->n_trace_unkns_per_face_dof() + i_unkn;
      dof_indices[idx1] = this->unkns_id_in_all_ranks[i_face][i_unkn];
    }
  //
  for (unsigned i_dof = 0; i_dof < dof_indices.size(); ++i_dof)
  {
    rho_n_vec = lu_of_Mat1.solve(Mat2.col(i_dof));
    q_n_vec = A_inv * (B1 * rho_n_vec - C1.col(i_dof));
    Eigen::VectorXd jth_col =
      C1.transpose() * q_n_vec + E1.transpose() * rho_n_vec + H.col(i_dof);
    int i_col = dof_indices[i_dof];
    in_solver->push_to_global_mat(dof_indices.data(), &i_col, jth_col,
                                  ADD_VALUES);
  }
  //
  rho_n_vec = lu_of_Mat1.solve(Fn - B1.transpose() * A_inv * R + E1 * gD_phi);
  q_n_vec = A_inv * (R + B1 * rho_n_vec);
  Eigen::MatrixXd jth_col =
    L - C1.transpose() * q_n_vec - E1.transpose() * rho_n_vec - H * gD_phi;
  in_solver->push_to_rhs_vec(dof_indices.data(), jth_col, ADD_VALUES);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::compute_my_local_unkns(const double *trace_sol)
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  unsigned n_trace_unkns = my_basis->n_trace_unkns_per_face_dof();
  compute_my_matrices();
  //
  double mu_n = own_cell->my_data->mu_n();
  double c_n =
    own_cell->my_data->alpha_n() * mu_n * own_cell->my_data->lambda_inv2_S();
  double mu_p = own_cell->my_data->mu_p();
  double c_p =
    own_cell->my_data->alpha_p() * mu_p * own_cell->my_data->lambda_inv2_S();
  double mu_r = own_cell->my_data->mu_r();
  double c_r =
    own_cell->my_data->alpha_r() * mu_r * own_cell->my_data->lambda_inv2_E();
  double mu_o = own_cell->my_data->mu_o();
  double c_o =
    own_cell->my_data->alpha_o() * mu_o * own_cell->my_data->lambda_inv2_E();
  //
  Eigen::MatrixXd A_inv = A1.inverse();
  Eigen::FullPivLU<Eigen::MatrixXd> lu_of_Mat1;
  //
  // Computations for the rho_n dof.
  //
  lu_of_Mat1.compute(B1.transpose() * (mu_n * A_inv) * B1 + D1 - c_n * D2);
  Eigen::VectorXd trace_vec = gD_rho_n;
  //
  for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
  {
    unsigned n_unkns = this->unkns_id_in_this_rank[i_face].size();
    assert(n_trace_unkns == n_unkns / 4);
    for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
    {
      unsigned j_unkn = 0 * n_trace_unkns + i_unkn;
      if (this->unkns_id_in_this_rank[i_face][j_unkn] >= 0)
        trace_vec(i_face * n_unkns + i_unkn) =
          trace_sol[this->unkns_id_in_this_rank[i_face][j_unkn]];
    }
  }
  rho_n_vec = lu_of_Mat1.solve(
    Fn + (B1.transpose() * (mu_n * A_inv) * C1 + E1 - c_n * E2) * trace_vec);
  q_n_vec = (mu_n * A_inv) * (B1 * rho_n_vec - C1 * trace_vec);
  //
  // Computations for the rho_p dof.
  //
  lu_of_Mat1.compute(B1.transpose() * (mu_p * A_inv) * B1 + D1 - c_p * D2);
  trace_vec = gD_rho_p;
  //
  for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
  {
    unsigned n_unkns = this->unkns_id_in_this_rank[i_face].size();
    assert(n_trace_unkns == n_unkns / 4);
    for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
    {
      unsigned j_unkn = 1 * n_trace_unkns + i_unkn;
      if (this->unkns_id_in_this_rank[i_face][j_unkn] >= 0)
        trace_vec(i_face * n_unkns + i_unkn) =
          trace_sol[this->unkns_id_in_this_rank[i_face][j_unkn]];
    }
  }
  rho_p_vec = lu_of_Mat1.solve(
    Fp + (B1.transpose() * (mu_p * A_inv) * C1 + E1 - c_p * E2) * trace_vec);
  q_p_vec = (mu_p * A_inv) * (B1 * rho_p_vec - C1 * trace_vec);
  //
  // Computations for the rho_r dof.
  //
  lu_of_Mat1.compute(B1.transpose() * (mu_r * A_inv) * B1 + D1 - c_r * D2);
  trace_vec = gD_rho_r;
  //
  for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
  {
    unsigned n_unkns = this->unkns_id_in_this_rank[i_face].size();
    assert(n_trace_unkns == n_unkns / 4);
    for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
    {
      unsigned j_unkn = 2 * n_trace_unkns + i_unkn;
      if (this->unkns_id_in_this_rank[i_face][j_unkn] >= 0)
        trace_vec(i_face * n_unkns + i_unkn) =
          trace_sol[this->unkns_id_in_this_rank[i_face][j_unkn]];
    }
  }
  rho_r_vec = lu_of_Mat1.solve(
    Fr + (B1.transpose() * (mu_r * A_inv) * C1 + E1 - c_r * E2) * trace_vec);
  q_r_vec = (mu_r * A_inv) * (B1 * rho_r_vec - C1 * trace_vec);
  //
  // Computations for the rho_o dof.
  //
  lu_of_Mat1.compute(B1.transpose() * (mu_o * A_inv) * B1 + D1 - c_o * D2);
  trace_vec = gD_rho_o;
  //
  for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
  {
    unsigned n_unkns = this->unkns_id_in_this_rank[i_face].size();
    assert(n_trace_unkns == n_unkns / 4);
    for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
    {
      unsigned j_unkn = 3 * n_trace_unkns + i_unkn;
      if (this->unkns_id_in_this_rank[i_face][j_unkn] >= 0)
        trace_vec(i_face * n_unkns + i_unkn) =
          trace_sol[this->unkns_id_in_this_rank[i_face][j_unkn]];
    }
  }
  rho_o_vec = lu_of_Mat1.solve(
    Fo + (B1.transpose() * (mu_o * A_inv) * C1 + E1 - c_o * E2) * trace_vec);
  q_o_vec = (mu_o * A_inv) * (B1 * rho_o_vec - C1 * trace_vec);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::compute_my_matrices()
{
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  unsigned n_trace_unkns = my_basis->n_trace_unkns_per_cell_dof();
  unsigned cell_quad_size = my_basis->get_cell_quad_size();
  unsigned face_quad_size = my_basis->get_face_quad_size();
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  //
  dealii::FEValuesExtractors::Scalar scalar(0);
  dealii::FEValuesExtractors::Vector fluxes(1);
  //
  my_basis->local_fe_val_in_cell->reinit(this->my_dealii_local_dofs_cell);
  //
  A1 = Eigen::MatrixXd::Zero(dim * n_scalar_unkns, dim * n_scalar_unkns);
  B1 = Eigen::MatrixXd::Zero(dim * n_scalar_unkns, n_scalar_unkns);
  C1 = Eigen::MatrixXd::Zero(dim * n_scalar_unkns, n_trace_unkns);
  D1 = Eigen::MatrixXd::Zero(n_scalar_unkns, n_scalar_unkns);
  D2 = Eigen::MatrixXd::Zero(n_scalar_unkns, n_scalar_unkns);
  E1 = Eigen::MatrixXd::Zero(n_scalar_unkns, n_trace_unkns);
  E2 = Eigen::MatrixXd::Zero(n_scalar_unkns, n_trace_unkns);
  Fn = Eigen::VectorXd::Zero(n_scalar_unkns);
  Fp = Eigen::VectorXd::Zero(n_scalar_unkns);
  Fr = Eigen::VectorXd::Zero(n_scalar_unkns);
  Fo = Eigen::VectorXd::Zero(n_scalar_unkns);
  H = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  L = Eigen::VectorXd::Zero(n_trace_unkns);
  R = Eigen::VectorXd::Zero(n_trace_unkns);
  //
  for (unsigned i_quad = 0; i_quad < cell_quad_size; ++i_quad)
  {
    double JxW = my_basis->local_fe_val_in_cell->JxW(i_quad);
    for (unsigned i1 = n_scalar_unkns; i1 < (dim + 1) * n_scalar_unkns; ++i1)
    {
      dealii::Tensor<1, dim> q_i1 =
        (*my_basis->local_fe_val_in_cell)[fluxes].value(i1, i_quad);
      for (unsigned j1 = n_scalar_unkns; j1 < (dim + 1) * n_scalar_unkns; ++j1)
      {
        dealii::Tensor<1, dim> v_j1 =
          (*my_basis->local_fe_val_in_cell)[fluxes].value(j1, i_quad);
        A1(i1 - n_scalar_unkns, j1 - n_scalar_unkns) += q_i1 * v_j1 * JxW;
      }
    }
    //
    // We first obtain the value of E at the current quad point.
    //
    dealii::Tensor<1, dim> E_at_quad;
    double f_n_val_at_quad = 0;
    double f_p_val_at_quad = 0;
    double f_r_val_at_quad = 0;
    double f_o_val_at_quad = 0;
    for (unsigned j1 = 0; j1 < n_scalar_unkns; ++j1)
    {
      double u_j1 = (*my_basis->local_fe_val_in_cell)[scalar].value(j1, i_quad);
      E_at_quad += u_j1 * E_vec[j1];
      f_n_val_at_quad += u_j1 * f_n_vec[j1];
      f_p_val_at_quad += u_j1 * f_p_vec[j1];
      f_r_val_at_quad += u_j1 * f_r_vec[j1];
      f_o_val_at_quad += u_j1 * f_o_vec[j1];
    }
    //
    for (unsigned i1 = 0; i1 < n_scalar_unkns; ++i1)
    {
      double u_i1 = (*my_basis->local_fe_val_in_cell)[scalar].value(i1, i_quad);
      Fn(i1) += u_i1 * JxW * f_n_val_at_quad;
      Fp(i1) += u_i1 * JxW * f_p_val_at_quad;
      Fr(i1) += u_i1 * JxW * f_r_val_at_quad;
      Fo(i1) += u_i1 * JxW * f_o_val_at_quad;
    }
    //
    for (unsigned j1 = 0; j1 < n_scalar_unkns; ++j1)
    {
      double u_j1 = (*my_basis->local_fe_val_in_cell)[scalar].value(j1, i_quad);
      for (unsigned i1 = n_scalar_unkns; i1 < (dim + 1) * n_scalar_unkns; ++i1)
      {
        double q_i1_div =
          (*my_basis->local_fe_val_in_cell)[fluxes].divergence(i1, i_quad);
        B1(i1 - n_scalar_unkns, j1) += u_j1 * q_i1_div * JxW;
      }
      for (unsigned i1 = 0; i1 < n_scalar_unkns; ++i1)
      {
        dealii::Tensor<1, dim> grad_u_i1 =
          (*my_basis->local_fe_val_in_cell)[scalar].gradient(i1, i_quad);
        D2(i1, j1) += grad_u_i1 * E_at_quad * u_j1 * JxW;
      }
    }
  }
  //
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    unsigned i_half = this->half_range_flag[i_face];
    dealii::FEFaceValues<dim> *fe_face_val =
      my_basis->trace_fe_face_val[i_half].get();
    fe_face_val->reinit(this->my_dealii_trace_dofs_cell, i_face);
    my_basis->local_fe_val_on_faces[i_face]->reinit(
      this->my_dealii_local_dofs_cell);
    std::vector<dealii::Point<spacedim> > face_quad_locs =
      fe_face_val->get_quadrature_points();
    // Loop 1
    for (unsigned i_face_quad = 0; i_face_quad < face_quad_size; ++i_face_quad)
    {
      double face_JxW = fe_face_val->JxW(i_face_quad);
      dealii::Tensor<1, dim> n_vec = fe_face_val->normal_vector(i_face_quad);
      //
      // We obtain gN.n and E*.n at face quad point.
      //
      dealii::Tensor<1, dim> E_star_at_face_quad;
      dealii::Tensor<1, dim> gN_at_face_quad;
      for (unsigned j_face_unkn = 0; j_face_unkn < n_trace_unkns; ++j_face_unkn)
      {
        double lambda_j1 = fe_face_val->shape_value(j_face_unkn, i_face_quad);
        gN_at_face_quad += gN_rho_n[j_face_unkn] * lambda_j1;
        E_star_at_face_quad += E_star_vec[j_face_unkn] * lambda_j1;
      }
      double gN_dot_n_at_face_quad = gN_at_face_quad * n_vec;
      double E_star_dot_n_at_face_quad = E_star_at_face_quad * n_vec;
      //
      // Then tau is obtained according to E*.
      //
      const double tau_at_quad =
        own_cell->my_data->tau(face_quad_locs[i_face_quad]);
      //
      for (unsigned i1 = 0; i1 < n_scalar_unkns; ++i1)
      {
        double u_i1 = (*my_basis->local_fe_val_on_faces[i_face])[scalar].value(
          i1, i_face_quad);
        for (unsigned j1 = 0; j1 < n_scalar_unkns; ++j1)
        {
          double w_j1 =
            (*my_basis->local_fe_val_on_faces[i_face])[scalar].value(
              j1, i_face_quad);
          D1(i1, j1) += face_JxW * tau_at_quad * u_i1 * w_j1;
        }
      }
      // Loop 2
      for (unsigned i_face_unkn = 0; i_face_unkn < n_trace_unkns; ++i_face_unkn)
      {
        double lambda_i1 = fe_face_val->shape_value(i_face_unkn, i_face_quad);
        for (unsigned j1 = n_scalar_unkns; j1 < (dim + 1) * n_scalar_unkns;
             ++j1)
        {
          dealii::Tensor<1, dim> v_j1 =
            (*my_basis->local_fe_val_on_faces[i_face])[fluxes].value(
              j1, i_face_quad);
          C1(j1 - n_scalar_unkns, i_face_unkn) +=
            face_JxW * lambda_i1 * (v_j1 * n_vec);
        }
        //
        for (unsigned j1 = 0; j1 < n_scalar_unkns; ++j1)
        {
          double w_j1 =
            (*my_basis->local_fe_val_on_faces[i_face])[scalar].value(
              j1, i_face_quad);
          E1(j1, i_face_unkn) += w_j1 * tau_at_quad * lambda_i1 * face_JxW;
          E2(j1, i_face_unkn) +=
            w_j1 * E_star_dot_n_at_face_quad * lambda_i1 * face_JxW;
        }
        //
        for (unsigned j_face_unkn = 0; j_face_unkn < n_trace_unkns;
             ++j_face_unkn)
        {
          double lambda_j1 = fe_face_val->shape_value(j_face_unkn, i_face_quad);
          H(i_face_unkn, j_face_unkn) -=
            lambda_i1 * tau_at_quad * lambda_j1 * face_JxW;
        }
        L(i_face_quad) = lambda_i1 * face_JxW * gN_dot_n_at_face_quad;
      }
      // Loop 2
    }
    // Loop 1
  }
  R = -C1 * gD_rho_n;
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::interpolate_to_my_interior()
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  exact_rho_n.resize(n_scalar_unkns);
  exact_q_n.resize(n_scalar_unkns * dim);
  exact_rho_p.resize(n_scalar_unkns);
  exact_q_p.resize(n_scalar_unkns * dim);
  exact_rho_r.resize(n_scalar_unkns);
  exact_q_r.resize(n_scalar_unkns * dim);
  exact_rho_o.resize(n_scalar_unkns);
  exact_q_o.resize(n_scalar_unkns * dim);
  //
  my_basis->local_fe_val_at_cell_supp->reinit(this->my_dealii_local_dofs_cell);
  //
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    dealii::Point<spacedim> q_point =
      my_basis->local_fe_val_at_cell_supp->quadrature_point(i_unkn);
    exact_rho_n(i_unkn) = own_cell->my_data->exact_rho_n(q_point);
    exact_rho_p(i_unkn) = own_cell->my_data->exact_rho_p(q_point);
    exact_rho_r(i_unkn) = own_cell->my_data->exact_rho_r(q_point);
    exact_rho_o(i_unkn) = own_cell->my_data->exact_rho_o(q_point);
    //
    dealii::Tensor<1, dim> q_val = own_cell->my_data->exact_q_n(q_point);
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      exact_q_n(i_dim * n_scalar_unkns + i_unkn) = q_val[i_dim];
    q_val = own_cell->my_data->exact_q_p(q_point);
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      exact_q_p(i_dim * n_scalar_unkns + i_unkn) = q_val[i_dim];
    q_val = own_cell->my_data->exact_q_r(q_point);
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      exact_q_r(i_dim * n_scalar_unkns + i_unkn) = q_val[i_dim];
    q_val = own_cell->my_data->exact_q_o(q_point);
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      exact_q_o(i_dim * n_scalar_unkns + i_unkn) = q_val[i_dim];
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::fill_my_viz_vector(distributed_vector<dim, spacedim> *out_vec)
{
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  //
  unsigned offset1 = 0;
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    int idx1 = this->local_interior_unkn_idx[offset1 + i_unkn];
    out_vec->assemble(idx1, rho_n_vec(i_unkn));
  }
  for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
  {
    unsigned offset2 = (i_dim + 1) * n_scalar_unkns + offset1;
    for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
    {
      int idx2 = this->local_interior_unkn_idx[offset2 + i_unkn];
      out_vec->assemble(idx2, q_n_vec(i_dim * n_scalar_unkns + i_unkn));
    }
  }
  //
  offset1 = (dim + 1) * n_scalar_unkns;
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    int idx1 = this->local_interior_unkn_idx[offset1 + i_unkn];
    out_vec->assemble(idx1, rho_p_vec(i_unkn));
  }
  for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
  {
    unsigned offset2 = (i_dim + 1) * n_scalar_unkns + offset1;
    for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
    {
      int idx2 = this->local_interior_unkn_idx[offset2 + i_unkn];
      out_vec->assemble(idx2, q_p_vec(i_dim * n_scalar_unkns + i_unkn));
    }
  }
  //
  offset1 = 2 * (dim + 1) * n_scalar_unkns;
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    int idx1 = this->local_interior_unkn_idx[offset1 + i_unkn];
    out_vec->assemble(idx1, rho_r_vec(i_unkn));
  }
  for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
  {
    unsigned offset2 = (i_dim + 1) * n_scalar_unkns + offset1;
    for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
    {
      int idx2 = this->local_interior_unkn_idx[offset2 + i_unkn];
      out_vec->assemble(idx2, q_r_vec(i_dim * n_scalar_unkns + i_unkn));
    }
  }
  //
  offset1 = 3 * (dim + 1) * n_scalar_unkns;
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    int idx1 = this->local_interior_unkn_idx[offset1 + i_unkn];
    out_vec->assemble(idx1, rho_o_vec(i_unkn));
  }
  for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
  {
    unsigned offset2 = (i_dim + 1) * n_scalar_unkns + offset1;
    for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
    {
      int idx2 = this->local_interior_unkn_idx[offset2 + i_unkn];
      out_vec->assemble(idx2, q_o_vec(i_dim * n_scalar_unkns + i_unkn));
    }
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::fill_my_refn_vector(distributed_vector<dim, spacedim> *out_vec)
{
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  //
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    int idx1 = this->my_cell->id_num * n_scalar_unkns + i_unkn;
    out_vec->assemble(idx1, rho_n_vec(i_unkn));
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::set_my_source_and_BCs()
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  //
  // ***** To be checked later *****
  //
  E_vec.resize(n_scalar_unkns);
  //
  f_n_vec = Eigen::VectorXd::Zero(n_scalar_unkns);
  f_p_vec = Eigen::VectorXd::Zero(n_scalar_unkns);
  f_r_vec = Eigen::VectorXd::Zero(n_scalar_unkns);
  f_o_vec = Eigen::VectorXd::Zero(n_scalar_unkns);
  //
  my_basis->local_fe_val_at_cell_supp->reinit(this->my_dealii_local_dofs_cell);
  //
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    dealii::Point<spacedim> q_point =
      my_basis->local_fe_val_at_cell_supp->quadrature_point(i_unkn);
    E_vec[i_unkn] = own_cell->my_data->electric_field(q_point);
    f_n_vec[i_unkn] = own_cell->my_data->rho_n_rhs_func(q_point);
    f_p_vec[i_unkn] = own_cell->my_data->rho_p_rhs_func(q_point);
    f_r_vec[i_unkn] = own_cell->my_data->rho_r_rhs_func(q_point);
    f_o_vec[i_unkn] = own_cell->my_data->rho_o_rhs_func(q_point);
  }
  //
  // ***** To be checked later *****
  //
  E_star_vec.resize(my_basis->n_trace_unkns_per_cell_dof());
  //
  gD_rho_n = Eigen::VectorXd::Zero(my_basis->n_trace_unkns_per_cell_dof());
  gD_rho_p = Eigen::VectorXd::Zero(my_basis->n_trace_unkns_per_cell_dof());
  gD_rho_r = Eigen::VectorXd::Zero(my_basis->n_trace_unkns_per_cell_dof());
  gD_rho_o = Eigen::VectorXd::Zero(my_basis->n_trace_unkns_per_cell_dof());
  gN_rho_n.resize(my_basis->n_trace_unkns_per_cell_dof());
  gN_rho_p.resize(my_basis->n_trace_unkns_per_cell_dof());
  gN_rho_r.resize(my_basis->n_trace_unkns_per_cell_dof());
  gN_rho_o.resize(my_basis->n_trace_unkns_per_cell_dof());
  unsigned n_dofs_per_face = my_basis->n_trace_unkns_per_face_dof();
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    my_basis->trace_fe_face_val_at_supp->reinit(this->my_dealii_trace_dofs_cell,
                                                i_face);
    const std::vector<dealii::Point<spacedim> > &face_supp_locs =
      my_basis->trace_fe_face_val_at_supp->get_quadrature_points();
    unsigned idx1 = i_face * n_dofs_per_face;
    for (unsigned i1 = 0; i1 < n_dofs_per_face; ++i1)
    {
      E_star_vec[idx1 + i1] =
        own_cell->my_data->electric_field(face_supp_locs[i1]);
      //
      double gD_at_face_supp = own_cell->my_data->gD_rho_n(face_supp_locs[i1]);
      dealii::Tensor<1, dim> gN_at_face_supp =
        own_cell->my_data->gN_rho_n(face_supp_locs[i1]);
      if (this->BCs[i_face] == boundary_condition::essential)
        gD_rho_n(idx1 + i1) = gD_at_face_supp;
      if (this->BCs[i_face] == boundary_condition::natural)
        gN_rho_n[idx1 + i1] = gN_at_face_supp;
      //
      gD_at_face_supp = own_cell->my_data->gD_rho_p(face_supp_locs[i1]);
      gN_at_face_supp = own_cell->my_data->gN_rho_p(face_supp_locs[i1]);
      if (this->BCs[i_face] == boundary_condition::essential)
        gD_rho_p(idx1 + i1) = gD_at_face_supp;
      if (this->BCs[i_face] == boundary_condition::natural)
        gN_rho_p[idx1 + i1] = gN_at_face_supp;
      //
      gD_at_face_supp = own_cell->my_data->gD_rho_r(face_supp_locs[i1]);
      gN_at_face_supp = own_cell->my_data->gN_rho_r(face_supp_locs[i1]);
      if (this->BCs[i_face] == boundary_condition::essential)
        gD_rho_r(idx1 + i1) = gD_at_face_supp;
      if (this->BCs[i_face] == boundary_condition::natural)
        gN_rho_r[idx1 + i1] = gN_at_face_supp;
      //
      gD_at_face_supp = own_cell->my_data->gD_rho_o(face_supp_locs[i1]);
      gN_at_face_supp = own_cell->my_data->gN_rho_o(face_supp_locs[i1]);
      if (this->BCs[i_face] == boundary_condition::essential)
        gD_rho_o(idx1 + i1) = gD_at_face_supp;
      if (this->BCs[i_face] == boundary_condition::natural)
        gN_rho_o[idx1 + i1] = gN_at_face_supp;
    }
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::compute_my_errors(std::vector<double> *sum_of_L2_errors)
{
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  //
  my_basis->local_fe_val_in_cell->reinit(this->my_dealii_local_dofs_cell);
  unsigned cell_quad_size =
    my_basis->local_fe_val_in_cell->get_quadrature().size();
  //
  for (unsigned i_quad = 0; i_quad < cell_quad_size; ++i_quad)
  {
    double JxW = my_basis->local_fe_val_in_cell->JxW(i_quad);
    //
    double rho_n_error_at_quad = 0;
    double rho_p_error_at_quad = 0;
    double rho_r_error_at_quad = 0;
    double rho_o_error_at_quad = 0;
    dealii::Tensor<1, dim> q_n_error_at_quad, q_p_error_at_quad,
      q_r_error_at_quad, q_o_error_at_quad;
    for (unsigned i1 = 0; i1 < n_scalar_unkns; ++i1)
    {
      double shape_val_u =
        my_basis->local_fe_val_in_cell->shape_value(i1, i_quad);
      rho_n_error_at_quad += (exact_rho_n(i1) - rho_n_vec(i1)) * shape_val_u;
      rho_p_error_at_quad += (exact_rho_p(i1) - rho_p_vec(i1)) * shape_val_u;
      rho_r_error_at_quad += (exact_rho_r(i1) - rho_r_vec(i1)) * shape_val_u;
      rho_o_error_at_quad += (exact_rho_o(i1) - rho_o_vec(i1)) * shape_val_u;
      for (unsigned j1 = 0; j1 < dim; ++j1)
      {
        unsigned i2 = j1 * n_scalar_unkns + i1;
        q_n_error_at_quad[j1] += shape_val_u * (exact_q_n(i2) - q_n_vec(i2));
        q_p_error_at_quad[j1] += shape_val_u * (exact_q_p(i2) - q_p_vec(i2));
        q_r_error_at_quad[j1] += shape_val_u * (exact_q_r(i2) - q_r_vec(i2));
        q_o_error_at_quad[j1] += shape_val_u * (exact_q_o(i2) - q_o_vec(i2));
      }
    }
    (*sum_of_L2_errors)[0] += JxW * rho_n_error_at_quad * rho_n_error_at_quad;
    (*sum_of_L2_errors)[1] += JxW * q_n_error_at_quad * q_n_error_at_quad;
    (*sum_of_L2_errors)[2] += JxW * rho_p_error_at_quad * rho_p_error_at_quad;
    (*sum_of_L2_errors)[3] += JxW * q_p_error_at_quad * q_p_error_at_quad;
    (*sum_of_L2_errors)[4] += JxW * rho_r_error_at_quad * rho_r_error_at_quad;
    (*sum_of_L2_errors)[5] += JxW * q_r_error_at_quad * q_r_error_at_quad;
    (*sum_of_L2_errors)[6] += JxW * rho_o_error_at_quad * rho_o_error_at_quad;
    (*sum_of_L2_errors)[7] += JxW * q_o_error_at_quad * q_o_error_at_quad;
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::assign_BCs(reactive_interface *in_cell, BC_Func func)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->assign_my_BCs(func);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::interpolate_to_interior(reactive_interface *in_cell)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->interpolate_to_my_interior();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::fill_viz_vector(reactive_interface *in_cell,
                              distributed_vector<dim, spacedim> *out_vec)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->fill_my_viz_vector(out_vec);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::fill_refn_vector(reactive_interface *in_cell,
                               distributed_vector<dim, spacedim> *out_vec)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->fill_my_refn_vector(out_vec);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::set_source_and_BCs(reactive_interface *in_cell)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->set_my_source_and_BCs();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<BasisType>::
  assemble_globals(reactive_interface *in_cell,
                   solvers::base_implicit_solver<dim, spacedim> *in_solver)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->assemble_my_globals(in_solver);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::compute_local_unkns(reactive_interface *in_cell,
                                  const double *trace_sol)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->compute_my_local_unkns(trace_sol);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::compute_errors(reactive_interface *in_cell,
                             std::vector<double> *sum_of_L2_errors)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->compute_my_errors(sum_of_L2_errors);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::visualize_results(const viz_data &in_viz_data)
{
  unsigned time_level = 0;
  const auto &tria = in_viz_data.my_dof_handler->get_triangulation();
  unsigned n_active_cells = tria.n_active_cells();
  //
  int comm_rank, comm_size;
  MPI_Comm_rank(in_viz_data.my_comm, &comm_rank);
  MPI_Comm_size(in_viz_data.my_comm, &comm_size);
  //
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(*in_viz_data.my_dof_handler);
  //
  std::vector<std::string> solution_names(4 * (dim + 1));
  for (unsigned i1 = 0; i1 < 4; ++i1)
    solution_names[(dim + 1) * i1] = in_viz_data.my_var_names[2 * i1];
  for (unsigned j1 = 0; j1 < 4; ++j1)
    for (unsigned i1 = 1; i1 < dim + 1; ++i1)
      solution_names[(dim + 1) * j1 + i1] =
        in_viz_data.my_var_names[2 * j1 + 1];
  //
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation;
  data_component_interpretation.reserve(4 * (dim + 1));
  for (unsigned j1 = 0; j1 < 4; ++j1)
  {
    data_component_interpretation.push_back(
      dealii::DataComponentInterpretation::component_is_scalar);
    for (unsigned i1 = 0; i1 < dim; ++i1)
      data_component_interpretation.push_back(
        dealii::DataComponentInterpretation::component_is_part_of_vector);
  }
  //
  data_out.add_data_vector(*in_viz_data.my_viz_sol,
                           solution_names,
                           dealii::DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  //
  dealii::Vector<float> subdomain(n_active_cells);
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = comm_rank;
  data_out.add_data_vector(subdomain, "subdomain");
  //
  data_out.build_patches();
  //
  const std::string filename =
    (in_viz_data.my_out_filename + "-" +
     dealii::Utilities::int_to_string(comm_rank, 4) + "-" +
     dealii::Utilities::int_to_string(time_level, 4));
  //
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);
  //
  if (comm_rank == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < comm_size; ++i)
      filenames.push_back(in_viz_data.my_out_filename + "-" +
                          dealii::Utilities::int_to_string(i, 4) + "-" +
                          dealii::Utilities::int_to_string(time_level, 4) +
                          ".vtu");
    std::ofstream master_output((filename + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}
