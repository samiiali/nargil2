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

template <int dim, int spacedim>
template <typename OtherCellEq>
void nargil::reactive_interface<dim, spacedim>::connect_to_other_cell(
  OtherCellEq *)
{
  assert(false);
}

//
//

namespace nargil
{
/**
 *
 * This is specialized template of the above class. In C++ there is little
 * chance to specialize a function template without specializing the class
 * template.
 *
 */
template <>
template <>
void reactive_interface<2, 2>::connect_to_other_cell(
  diffusion<2, 2> *in_relevant_cell)
{
  my_relevant_diff_cell = in_relevant_cell;
}
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
  unsigned n_trace_unkns = my_basis->n_trace_unkns_per_face_dof();
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
  Eigen::MatrixXd d_rho_vec, d_q_vec, NR_rhs;
  Eigen::MatrixXd A_inv = A1.inverse();
  Eigen::FullPivLU<Eigen::MatrixXd> lu_of_Mat1;
  //
  std::vector<int> indices_n(my_basis->n_trace_unkns_per_cell_dof());
  std::vector<int> indices_p(my_basis->n_trace_unkns_per_cell_dof());
  std::vector<int> indices_r(my_basis->n_trace_unkns_per_cell_dof());
  std::vector<int> indices_o(my_basis->n_trace_unkns_per_cell_dof());
  for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
  {
    for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
    {
      unsigned idx = i_face * n_trace_unkns + i_unkn;
      unsigned j_unkn0 = 0 * my_basis->n_trace_unkns_per_cell_dof() + i_unkn;
      unsigned j_unkn1 = 1 * my_basis->n_trace_unkns_per_face_dof() + i_unkn;
      unsigned j_unkn2 = 2 * my_basis->n_trace_unkns_per_face_dof() + i_unkn;
      unsigned j_unkn3 = 3 * my_basis->n_trace_unkns_per_face_dof() + i_unkn;
      indices_n[idx] = this->unkns_id_in_all_ranks[i_face][j_unkn0];
      indices_p[idx] = this->unkns_id_in_all_ranks[i_face][j_unkn1];
      indices_r[idx] = this->unkns_id_in_all_ranks[i_face][j_unkn2];
      indices_o[idx] = this->unkns_id_in_all_ranks[i_face][j_unkn3];
    }
  }
  //
  // Assembeling equations rho_n
  //
  if (this->local_equation_is_active[0])
  {
    lu_of_Mat1.compute(B1.transpose() * (mu_n * A_inv) * B1 + D1 - c_n * D2);
    Eigen::MatrixXd Mat2 =
      (B1.transpose() * (mu_n * A_inv) * C1 + E1 - c_n * E2);
    //
    for (unsigned i_dof = 0; i_dof < indices_n.size(); ++i_dof)
    {
      d_rho_vec = lu_of_Mat1.solve(Mat2.col(i_dof));
      d_q_vec = mu_n * A_inv * (B1 * d_rho_vec - C1.col(i_dof));
      Eigen::VectorXd jth_col_n = C1.transpose() * d_q_vec +
                                  E1.transpose() * d_rho_vec - H1.col(i_dof) +
                                  c_n * H2.col(i_dof) - H11.col(i_dof);
      int i_col_n = indices_n[i_dof];
      in_solver->push_to_global_mat(indices_n.data(), &i_col_n, jth_col_n,
                                    ADD_VALUES);
      //
      Eigen::VectorXd jth_col_o = -H14.col(i_dof);
      int i_col_o = indices_o[i_dof];
      in_solver->push_to_global_mat(indices_n.data(), &i_col_o, jth_col_o,
                                    ADD_VALUES);
    }
    //
    d_rho_vec = lu_of_Mat1.solve(Fn + B1.transpose() * mu_n * A_inv * Rn);
    d_q_vec = mu_n * A_inv * (-Rn + B1 * d_rho_vec);
    NR_rhs = Ln - C1.transpose() * d_q_vec - E1.transpose() * d_rho_vec;
    in_solver->push_to_rhs_vec(indices_n.data(), NR_rhs, ADD_VALUES);
  }
  //
  // Assembeling equations rho_p
  //
  if (this->local_equation_is_active[1])
  {
    lu_of_Mat1.compute(B1.transpose() * (mu_p * A_inv) * B1 + D1 - c_p * D2);
    Eigen::MatrixXd Mat2 =
      (B1.transpose() * (mu_p * A_inv) * C1 + E1 - c_p * E2);
    //
    for (unsigned i_dof = 0; i_dof < indices_p.size(); ++i_dof)
    {
      d_rho_vec = lu_of_Mat1.solve(Mat2.col(i_dof));
      d_q_vec = mu_p * A_inv * (B1 * d_rho_vec - C1.col(i_dof));
      Eigen::VectorXd jth_col_p = C1.transpose() * d_q_vec +
                                  E1.transpose() * d_rho_vec - H1.col(i_dof) +
                                  c_p * H2.col(i_dof) - H22.col(i_dof);
      int i_col_p = indices_p[i_dof];
      in_solver->push_to_global_mat(indices_p.data(), &i_col_p, jth_col_p,
                                    ADD_VALUES);
      //
      Eigen::VectorXd jth_col_r = -H23.col(i_dof);
      int i_col_r = indices_r[i_dof];
      in_solver->push_to_global_mat(indices_p.data(), &i_col_r, jth_col_r,
                                    ADD_VALUES);
    }
    //
    d_rho_vec = lu_of_Mat1.solve(Fp + B1.transpose() * mu_p * A_inv * Rp);
    d_q_vec = mu_p * A_inv * (-Rp + B1 * d_rho_vec);
    NR_rhs = Lp - C1.transpose() * d_q_vec - E1.transpose() * d_rho_vec;
    in_solver->push_to_rhs_vec(indices_p.data(), NR_rhs, ADD_VALUES);
  }
  //
  // Assembeling equations rho_r
  //
  if (this->local_equation_is_active[2])
  {
    lu_of_Mat1.compute(B1.transpose() * (mu_r * A_inv) * B1 + D1 - c_r * D2);
    Eigen::MatrixXd Mat2 =
      (B1.transpose() * (mu_r * A_inv) * C1 + E1 - c_r * E2);
    //
    for (unsigned i_dof = 0; i_dof < indices_r.size(); ++i_dof)
    {
      d_rho_vec = lu_of_Mat1.solve(Mat2.col(i_dof));
      d_q_vec = mu_r * A_inv * (B1 * d_rho_vec - C1.col(i_dof));
      Eigen::VectorXd jth_col_r = C1.transpose() * d_q_vec +
                                  E1.transpose() * d_rho_vec - H1.col(i_dof) +
                                  c_r * H2.col(i_dof) - H23.col(i_dof);
      int i_col_r = indices_r[i_dof];
      in_solver->push_to_global_mat(indices_r.data(), &i_col_r, jth_col_r,
                                    ADD_VALUES);
      //
      Eigen::VectorXd jth_col_n = H11.col(i_dof);
      int i_col_n = indices_n[i_dof];
      in_solver->push_to_global_mat(indices_r.data(), &i_col_n, jth_col_n,
                                    ADD_VALUES);
      //
      Eigen::VectorXd jth_col_p = -H22.col(i_dof);
      int i_col_p = indices_p[i_dof];
      in_solver->push_to_global_mat(indices_r.data(), &i_col_p, jth_col_p,
                                    ADD_VALUES);
      //
      Eigen::VectorXd jth_col_o = H14.col(i_dof);
      int i_col_o = indices_o[i_dof];
      in_solver->push_to_global_mat(indices_r.data(), &i_col_o, jth_col_o,
                                    ADD_VALUES);
    }
    //
    d_rho_vec = lu_of_Mat1.solve(Fr + B1.transpose() * mu_r * A_inv * Rr);
    d_q_vec = mu_r * A_inv * (-Rr + B1 * d_rho_vec);
    NR_rhs = Lr - C1.transpose() * d_q_vec - E1.transpose() * d_rho_vec;
    in_solver->push_to_rhs_vec(indices_r.data(), NR_rhs, ADD_VALUES);
  }
  //
  // Assembeling equations rho_o
  //
  if (this->local_equation_is_active[3])
  {
    lu_of_Mat1.compute(B1.transpose() * (mu_o * A_inv) * B1 + D1 - c_o * D2);
    Eigen::MatrixXd Mat2 =
      (B1.transpose() * (mu_o * A_inv) * C1 + E1 - c_o * E2);
    //
    for (unsigned i_dof = 0; i_dof < indices_o.size(); ++i_dof)
    {
      d_rho_vec = lu_of_Mat1.solve(Mat2.col(i_dof));
      d_q_vec = mu_o * A_inv * (B1 * d_rho_vec - C1.col(i_dof));
      Eigen::VectorXd jth_col_o = C1.transpose() * d_q_vec +
                                  E1.transpose() * d_rho_vec - H1.col(i_dof) +
                                  c_o * H2.col(i_dof) - H14.col(i_dof);
      int i_col_o = indices_o[i_dof];
      in_solver->push_to_global_mat(indices_o.data(), &i_col_o, jth_col_o,
                                    ADD_VALUES);
      //
      Eigen::VectorXd jth_col_n = -H11.col(i_dof);
      int i_col_n = indices_n[i_dof];
      in_solver->push_to_global_mat(indices_o.data(), &i_col_n, jth_col_n,
                                    ADD_VALUES);
      //
      Eigen::VectorXd jth_col_p = H22.col(i_dof);
      int i_col_p = indices_p[i_dof];
      in_solver->push_to_global_mat(indices_o.data(), &i_col_p, jth_col_p,
                                    ADD_VALUES);
      //
      Eigen::VectorXd jth_col_r = H23.col(i_dof);
      int i_col_r = indices_r[i_dof];
      in_solver->push_to_global_mat(indices_o.data(), &i_col_r, jth_col_r,
                                    ADD_VALUES);
    }
    //
    d_rho_vec = lu_of_Mat1.solve(Fo + B1.transpose() * mu_o * A_inv * Ro);
    d_q_vec = mu_o * A_inv * (-Ro + B1 * d_rho_vec);
    NR_rhs = Lo - C1.transpose() * d_q_vec - E1.transpose() * d_rho_vec;
    in_solver->push_to_rhs_vec(indices_o.data(), NR_rhs, ADD_VALUES);
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<BasisType>::
  assemble_my_trpz_globals(
    solvers::base_implicit_solver<dim, spacedim> *in_solver)
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  unsigned n_trace_unkns = my_basis->n_trace_unkns_per_face_dof();
  ode_solvers::trapezoidal_solver<Eigen::VectorXd> *own_time_integrator =
    static_cast<ode_solvers::trapezoidal_solver<Eigen::VectorXd> *>(
      my_time_integrator);
  double dt = own_time_integrator->my_h;
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
  Eigen::MatrixXd d_rho_vec, d_q_vec, NR_rhs;
  Eigen::MatrixXd A_inv = A1.inverse();
  Eigen::FullPivLU<Eigen::MatrixXd> lu_of_Mat3;
  //
  std::vector<int> indices_n(my_basis->n_trace_unkns_per_cell_dof());
  std::vector<int> indices_p(my_basis->n_trace_unkns_per_cell_dof());
  std::vector<int> indices_r(my_basis->n_trace_unkns_per_cell_dof());
  std::vector<int> indices_o(my_basis->n_trace_unkns_per_cell_dof());
  for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
  {
    for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
    {
      unsigned idx = i_face * n_trace_unkns + i_unkn;
      unsigned j_unkn0 = 0 * my_basis->n_trace_unkns_per_cell_dof() + i_unkn;
      unsigned j_unkn1 = 1 * my_basis->n_trace_unkns_per_face_dof() + i_unkn;
      unsigned j_unkn2 = 2 * my_basis->n_trace_unkns_per_face_dof() + i_unkn;
      unsigned j_unkn3 = 3 * my_basis->n_trace_unkns_per_face_dof() + i_unkn;
      indices_n[idx] = this->unkns_id_in_all_ranks[i_face][j_unkn0];
      indices_p[idx] = this->unkns_id_in_all_ranks[i_face][j_unkn1];
      indices_r[idx] = this->unkns_id_in_all_ranks[i_face][j_unkn2];
      indices_o[idx] = this->unkns_id_in_all_ranks[i_face][j_unkn3];
    }
  }
  //
  // Assembeling equations rho_n
  //
  if (this->local_equation_is_active[0])
  {
    lu_of_Mat3.compute(
      1 / 2. * (B1.transpose() * (mu_n * A_inv) * B1 + D1 - c_n * D2) +
      1.0 / dt * A1);
    Eigen::MatrixXd Mat4 =
      (B1.transpose() * (mu_n * A_inv) * C1 + E1 - c_n * E2);
    //
    for (unsigned i_dof = 0; i_dof < indices_n.size(); ++i_dof)
    {
      d_rho_vec = lu_of_Mat3.solve(Mat4.col(i_dof));
      d_q_vec = mu_n * A_inv * (B1 * d_rho_vec - C1.col(i_dof));
      Eigen::VectorXd jth_col_n = C1.transpose() * d_q_vec +
                                  E1.transpose() * d_rho_vec - H1.col(i_dof) +
                                  c_n * H2.col(i_dof) - H11.col(i_dof);
      int i_col_n = indices_n[i_dof];
      in_solver->push_to_global_mat(indices_n.data(), &i_col_n, jth_col_n,
                                    ADD_VALUES);
      //
      Eigen::VectorXd jth_col_o = -H14.col(i_dof);
      int i_col_o = indices_o[i_dof];
      in_solver->push_to_global_mat(indices_n.data(), &i_col_o, jth_col_o,
                                    ADD_VALUES);
    }
    //
    d_rho_vec = lu_of_Mat3.solve(Fn + B1.transpose() * mu_n * A_inv * Rn);
    d_q_vec = mu_n * A_inv * (-Rn + B1 * d_rho_vec);
    NR_rhs = Ln - C1.transpose() * d_q_vec - E1.transpose() * d_rho_vec;
    in_solver->push_to_rhs_vec(indices_n.data(), NR_rhs, ADD_VALUES);
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::compute_my_NR_increments()
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
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
  if (this->local_equation_is_active[0])
  {
    lu_of_Mat1.compute(B1.transpose() * (mu_n * A_inv) * B1 + D1 - c_n * D2);
    d_rho_n = lu_of_Mat1.solve(
      Fn + mu_n * B1.transpose() * A_inv * Rn +
      (B1.transpose() * (mu_n * A_inv) * C1 + E1 - c_n * E2) * d_rho_n_hat);
    d_q_n = (mu_n * A_inv) * (-Rn + B1 * d_rho_n - C1 * d_rho_n_hat);
    //
    // Now, we apply the increment on the rho_n_hat, rho_n , and q_n.
    //
    rho_n_vec += d_rho_n;
    q_n_vec += d_q_n;
  }
  if (trace_unkns_is_active[0])
    rho_n_hat += d_rho_n_hat;
  //
  // Computations for the rho_p dof.
  //
  if (this->local_equation_is_active[1])
  {
    lu_of_Mat1.compute(B1.transpose() * (mu_p * A_inv) * B1 + D1 - c_p * D2);
    d_rho_p = lu_of_Mat1.solve(
      Fp + mu_p * B1.transpose() * A_inv * Rp +
      (B1.transpose() * (mu_p * A_inv) * C1 + E1 - c_p * E2) * d_rho_p_hat);
    d_q_p = (mu_p * A_inv) * (-Rp + B1 * d_rho_p - C1 * d_rho_p_hat);
    //
    // Now, we apply the increment on the rho_p_hat, rho_p , and q_p.
    //
    rho_p_vec += d_rho_p;
    q_p_vec += d_q_p;
  }
  if (trace_unkns_is_active[1])
    rho_p_hat += d_rho_p_hat;
  //
  // Computations for the rho_r dof.
  //
  if (this->local_equation_is_active[2])
  {
    lu_of_Mat1.compute(B1.transpose() * (mu_r * A_inv) * B1 + D1 - c_r * D2);
    d_rho_r = lu_of_Mat1.solve(
      Fr + mu_r * B1.transpose() * A_inv * Rr +
      (B1.transpose() * (mu_r * A_inv) * C1 + E1 - c_r * E2) * d_rho_r_hat);
    d_q_r = (mu_r * A_inv) * (-Rr + B1 * d_rho_r - C1 * d_rho_r_hat);
    //
    // Now, we apply the increment on the rho_r_hat, rho_r , and q_r.
    //
    rho_r_vec += d_rho_r;
    q_r_vec += d_q_r;
  }
  if (trace_unkns_is_active[2])
    rho_r_hat += d_rho_r_hat;
  //
  // Computations for the rho_o dof.
  //
  if (this->local_equation_is_active[3])
  {
    lu_of_Mat1.compute(B1.transpose() * (mu_o * A_inv) * B1 + D1 - c_o * D2);
    d_rho_o = lu_of_Mat1.solve(
      Fo + mu_o * B1.transpose() * A_inv * Ro +
      (B1.transpose() * (mu_o * A_inv) * C1 + E1 - c_o * E2) * d_rho_o_hat);
    d_q_o = (mu_o * A_inv) * (-Ro + B1 * d_rho_o - C1 * d_rho_o_hat);
    //
    // Now, we apply the increment on the rho_o_hat, rho_o , and q_o.
    //
    rho_o_vec += d_rho_o;
    q_o_vec += d_q_o;
  }
  if (trace_unkns_is_active[3])
    rho_o_hat += d_rho_o_hat;
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::extract_my_NR_increment(const double *trace_sol)
{
  unsigned n_trace_unkns = my_basis->n_trace_unkns_per_face_dof();
  //
  // Extracting d_rho_n_hat
  //
  if (trace_unkns_is_active[0])
  {
    d_rho_n_hat = Eigen::VectorXd::Zero(my_basis->n_trace_unkns_per_cell_dof());
    //
    for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
    {
      for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
      {
        unsigned j_unkn = 0 * n_trace_unkns + i_unkn;
        if (this->unkns_id_in_this_rank[i_face][j_unkn] >= 0)
          d_rho_n_hat(i_face * n_trace_unkns + i_unkn) =
            trace_sol[this->unkns_id_in_this_rank[i_face][j_unkn]];
      }
    }
  }
  //
  // Extracting d_rho_p_hat
  //
  if (trace_unkns_is_active[1])
  {
    d_rho_p_hat = Eigen::VectorXd::Zero(my_basis->n_trace_unkns_per_cell_dof());
    //
    for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
    {
      for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
      {
        unsigned j_unkn = 1 * n_trace_unkns + i_unkn;
        if (this->unkns_id_in_this_rank[i_face][j_unkn] >= 0)
          d_rho_p_hat(i_face * n_trace_unkns + i_unkn) =
            trace_sol[this->unkns_id_in_this_rank[i_face][j_unkn]];
      }
    }
  }
  //
  // Extracting d_rho_r_hat
  //
  if (trace_unkns_is_active[2])
  {
    d_rho_r_hat = Eigen::VectorXd::Zero(my_basis->n_trace_unkns_per_cell_dof());
    //
    for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
    {
      for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
      {
        unsigned j_unkn = 2 * n_trace_unkns + i_unkn;
        if (this->unkns_id_in_this_rank[i_face][j_unkn] >= 0)
          d_rho_r_hat(i_face * n_trace_unkns + i_unkn) =
            trace_sol[this->unkns_id_in_this_rank[i_face][j_unkn]];
      }
    }
  }
  //
  // Extracting d_rho_o_hat
  //
  if (trace_unkns_is_active[3])
  {
    d_rho_o_hat = Eigen::VectorXd::Zero(my_basis->n_trace_unkns_per_cell_dof());
    //
    for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
    {
      for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
      {
        unsigned j_unkn = 3 * n_trace_unkns + i_unkn;
        if (this->unkns_id_in_this_rank[i_face][j_unkn] >= 0)
          d_rho_o_hat(i_face * n_trace_unkns + i_unkn) =
            trace_sol[this->unkns_id_in_this_rank[i_face][j_unkn]];
      }
    }
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::compute_my_linear_matrices()
{
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  unsigned n_trace_unkns = my_basis->n_trace_unkns_per_cell_dof();
  //
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
  H1 = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  H2 = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  //
  if (this->local_equation_is_active[0])
    Fn = Eigen::VectorXd::Zero(n_scalar_unkns);
  if (this->local_equation_is_active[1])
    Fp = Eigen::VectorXd::Zero(n_scalar_unkns);
  if (this->local_equation_is_active[2])
    Fr = Eigen::VectorXd::Zero(n_scalar_unkns);
  if (this->local_equation_is_active[3])
    Fo = Eigen::VectorXd::Zero(n_scalar_unkns);
  if (trace_unkns_is_active[0])
    Ln = Eigen::VectorXd::Zero(n_trace_unkns);
  if (trace_unkns_is_active[1])
    Lp = Eigen::VectorXd::Zero(n_trace_unkns);
  if (trace_unkns_is_active[2])
    Lr = Eigen::VectorXd::Zero(n_trace_unkns);
  if (trace_unkns_is_active[3])
    Lo = Eigen::VectorXd::Zero(n_trace_unkns);
  //
  // Rn, Rp, Rr, Ro are defined at the end of this function.
  //
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
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        E_at_quad[i_dim] += u_j1 * E_field[i_dim * n_scalar_unkns + j1];
      if (this->local_equation_is_active[0])
        f_n_val_at_quad += u_j1 * f_n_vec[j1];
      if (this->local_equation_is_active[1])
        f_p_val_at_quad += u_j1 * f_p_vec[j1];
      if (this->local_equation_is_active[2])
        f_r_val_at_quad += u_j1 * f_r_vec[j1];
      if (this->local_equation_is_active[3])
        f_o_val_at_quad += u_j1 * f_o_vec[j1];
    }
    //
    for (unsigned i1 = 0; i1 < n_scalar_unkns; ++i1)
    {
      double u_i1 = (*my_basis->local_fe_val_in_cell)[scalar].value(i1, i_quad);
      if (this->local_equation_is_active[0])
        Fn(i1) += u_i1 * JxW * f_n_val_at_quad;
      if (this->local_equation_is_active[1])
        Fp(i1) += u_i1 * JxW * f_p_val_at_quad;
      if (this->local_equation_is_active[2])
        Fr(i1) += u_i1 * JxW * f_r_val_at_quad;
      if (this->local_equation_is_active[3])
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
      double gN_n_at_face_quad = 0;
      double gN_p_at_face_quad = 0;
      double gN_r_at_face_quad = 0;
      double gN_o_at_face_quad = 0;
      double E_star_dot_n_at_face_quad = 0.;
      //
      for (unsigned j_face_unkn = 0; j_face_unkn < n_trace_unkns; ++j_face_unkn)
      {
        double lambda_j1 = fe_face_val->shape_value(j_face_unkn, i_face_quad);
        E_star_dot_n_at_face_quad += E_star_dot_n[j_face_unkn] * lambda_j1;
        if (trace_unkns_is_active[0])
          gN_n_at_face_quad += gN_rho_n[j_face_unkn] * lambda_j1;
        if (trace_unkns_is_active[1])
          gN_p_at_face_quad += gN_rho_p[j_face_unkn] * lambda_j1;
        if (trace_unkns_is_active[2])
          gN_r_at_face_quad += gN_rho_r[j_face_unkn] * lambda_j1;
        if (trace_unkns_is_active[3])
          gN_o_at_face_quad += gN_rho_o[j_face_unkn] * lambda_j1;
      }
      //
      // *** Then tau may be obtained according to E*.
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
          H1(i_face_unkn, j_face_unkn) +=
            lambda_i1 * tau_at_quad * lambda_j1 * face_JxW;
          H2(i_face_unkn, j_face_unkn) +=
            E_star_dot_n_at_face_quad * lambda_i1 * lambda_j1 * face_JxW;
        }
        if (trace_unkns_is_active[0])
          Ln(i_face_unkn) += lambda_i1 * face_JxW * gN_n_at_face_quad;
        if (trace_unkns_is_active[1])
          Lp(i_face_unkn) += lambda_i1 * face_JxW * gN_p_at_face_quad;
        if (trace_unkns_is_active[2])
          Lr(i_face_unkn) += lambda_i1 * face_JxW * gN_r_at_face_quad;
        if (trace_unkns_is_active[3])
          Lo(i_face_unkn) += lambda_i1 * face_JxW * gN_o_at_face_quad;
      }
      // Loop 2
    }
    // Loop 1
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::compute_my_nonlinear_matrices()
{
  unsigned n_trace_unkns = my_basis->n_trace_unkns_per_cell_dof();
  unsigned face_quad_size = my_basis->get_face_quad_size();
  //
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  //
  // We assume that, compute_my_linear_matrices() has been called before.
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
  // Setting the sizes of Q1, Q2
  //
  Q1 = Eigen::VectorXd::Zero(n_trace_unkns);
  Q2 = Eigen::VectorXd::Zero(n_trace_unkns);
  H11 = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  H14 = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  H22 = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  H23 = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  //
  //
  //
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    unsigned i_half = this->half_range_flag[i_face];
    dealii::FEFaceValues<dim> *fe_face_val =
      my_basis->trace_fe_face_val[i_half].get();
    fe_face_val->reinit(this->my_dealii_trace_dofs_cell, i_face);
    my_basis->local_fe_val_on_faces[i_face]->reinit(
      this->my_dealii_local_dofs_cell);
    bool i_am_on_sigma_s = BCs[i_face] & boundary_condition::semiconductor_R_I;
    bool i_am_on_sigma_e = BCs[i_face] & boundary_condition::electrolyte_R_I;
    bool i_am_on_sigma = i_am_on_sigma_e || i_am_on_sigma_s;
    if (i_am_on_sigma)
    {
      //
      // Loop 1
      for (unsigned i_face_quad = 0; i_face_quad < face_quad_size;
           ++i_face_quad)
      {
        double face_JxW = fe_face_val->JxW(i_face_quad);
        dealii::Tensor<1, dim> n_vec = fe_face_val->normal_vector(i_face_quad);
        (void)n_vec;
        //
        // We obtain rho_hat_n, rho_hat_p, rho_hat_r, rho_hat_o at the
        // quadrature point.
        //
        double rho_hat_n_at_quad = 0;
        double rho_hat_p_at_quad = 0;
        double rho_hat_r_at_quad = 0;
        double rho_hat_o_at_quad = 0;
        //
        for (unsigned j_face_unkn = 0; j_face_unkn < n_trace_unkns;
             ++j_face_unkn)
        {
          double lambda_j1 = fe_face_val->shape_value(j_face_unkn, i_face_quad);
          if (trace_unkns_is_active[0])
            rho_hat_n_at_quad += rho_n_hat[j_face_unkn] * lambda_j1;
          if (trace_unkns_is_active[1])
            rho_hat_p_at_quad += rho_p_hat[j_face_unkn] * lambda_j1;
          if (trace_unkns_is_active[2])
            rho_hat_r_at_quad += rho_r_hat[j_face_unkn] * lambda_j1;
          if (trace_unkns_is_active[3])
            rho_hat_o_at_quad += rho_o_hat[j_face_unkn] * lambda_j1;
        }
        //
        double Q1_at_quad = get_Q1(rho_hat_n_at_quad, rho_hat_o_at_quad);
        double Q2_at_quad = get_Q2(rho_hat_p_at_quad, rho_hat_r_at_quad);
        double d_Q1_d_n_at_quad = get_d_Q1_d_n(rho_hat_o_at_quad);
        double d_Q1_d_o_at_quad = get_d_Q1_d_o(rho_hat_n_at_quad);
        double d_Q2_d_p_at_quad = get_d_Q2_d_p(rho_hat_r_at_quad);
        double d_Q2_d_r_at_quad = get_d_Q2_d_r(rho_hat_p_at_quad);
        //
        if (trace_unkns_is_active[0] && trace_unkns_is_active[2])
        {
          for (unsigned i_face_unkn = 0; i_face_unkn < n_trace_unkns;
               ++i_face_unkn)
          {
            double lambda_i1 =
              fe_face_val->shape_value(i_face_unkn, i_face_quad);
            for (unsigned j_face_unkn = 0; j_face_unkn < n_trace_unkns;
                 ++j_face_unkn)
            {
              double lambda_j1 =
                fe_face_val->shape_value(j_face_unkn, i_face_quad);
              H11(i_face_unkn, j_face_unkn) +=
                lambda_i1 * lambda_j1 * face_JxW * d_Q1_d_n_at_quad;
              H14(i_face_unkn, j_face_unkn) +=
                lambda_i1 * lambda_j1 * face_JxW * d_Q1_d_o_at_quad;
              H22(i_face_unkn, j_face_unkn) +=
                lambda_i1 * lambda_j1 * face_JxW * d_Q2_d_p_at_quad;
              H23(i_face_unkn, j_face_unkn) +=
                lambda_i1 * lambda_j1 * face_JxW * d_Q2_d_r_at_quad;
            }
            Q1(i_face_unkn) += lambda_i1 * face_JxW * Q1_at_quad;
            Q2(i_face_unkn) += lambda_i1 * face_JxW * Q2_at_quad;
          }
        }
      }
      // Loop 1
    }
  }
  //
  if (this->local_equation_is_active[0])
  {
    Rn = 1. / mu_n * A1 * q_n_vec - B1 * rho_n_vec + C1 * rho_n_hat;
    Fn += -B1.transpose() * q_n_vec - D1 * rho_n_vec + c_n * D2 * rho_n_vec +
          E1 * rho_n_hat - c_n * E2 * rho_n_hat;
    Ln += -C1.transpose() * q_n_vec - E1.transpose() * rho_n_vec +
          H1 * rho_n_hat - c_n * H2 * rho_n_hat;
  }
  if (this->trace_unkns_is_active[0])
    Ln += Q1;
  if (this->local_equation_is_active[1])
  {
    Rp = 1. / mu_p * A1 * q_p_vec - B1 * rho_p_vec + C1 * rho_p_hat;
    Fp += -B1.transpose() * q_p_vec - D1 * rho_p_vec + c_p * D2 * rho_p_vec +
          E1 * rho_p_hat - c_p * E2 * rho_p_hat;
    Lp += -C1.transpose() * q_p_vec - E1.transpose() * rho_p_vec +
          H1 * rho_p_hat - c_p * H2 * rho_p_hat;
  }
  if (this->trace_unkns_is_active[1])
    Lp += Q2;
  if (this->local_equation_is_active[2])
  {
    Rr = 1. / mu_r * A1 * q_r_vec - B1 * rho_r_vec + C1 * rho_r_hat;
    Fr += -B1.transpose() * q_r_vec - D1 * rho_r_vec + c_r * D2 * rho_r_vec +
          E1 * rho_r_hat - c_r * E2 * rho_r_hat;
    Lr += -C1.transpose() * q_r_vec - E1.transpose() * rho_r_vec +
          H1 * rho_r_hat - c_r * H2 * rho_r_hat;
  }
  if (this->trace_unkns_is_active[2])
    Lr += -Q1 + Q2;
  if (this->local_equation_is_active[3])
  {
    Ro = 1. / mu_o * A1 * q_o_vec - B1 * rho_o_vec + C1 * rho_o_hat;
    Fo += -B1.transpose() * q_o_vec - D1 * rho_o_vec + c_o * D2 * rho_o_vec +
          E1 * rho_o_hat - c_o * E2 * rho_o_hat;
    Lo += -C1.transpose() * q_o_vec - E1.transpose() * rho_o_vec +
          H1 * rho_o_hat - c_o * H2 * rho_o_hat;
  }
  if (this->trace_unkns_is_active[3])
    Lo += Q1 - Q2;
}

//
//

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::compute_my_nonlinear_trpz_matrices()
{
  unsigned n_trace_unkns = my_basis->n_trace_unkns_per_cell_dof();
  unsigned face_quad_size = my_basis->get_face_quad_size();
  //
  ode_solvers::trapezoidal_solver<Eigen::VectorXd> *own_time_integrator =
    static_cast<ode_solvers::trapezoidal_solver<Eigen::VectorXd> *>(
      my_time_integrator);
  double dt = own_time_integrator->my_h;
  //
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  //
  // We assume that, compute_my_linear_matrices() has been called before.
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
  // Setting the sizes of Q1, Q2
  //
  Q1 = Eigen::VectorXd::Zero(n_trace_unkns);
  Q2 = Eigen::VectorXd::Zero(n_trace_unkns);
  H11 = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  H14 = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  H22 = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  H23 = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  //
  //
  //
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    unsigned i_half = this->half_range_flag[i_face];
    dealii::FEFaceValues<dim> *fe_face_val =
      my_basis->trace_fe_face_val[i_half].get();
    fe_face_val->reinit(this->my_dealii_trace_dofs_cell, i_face);
    my_basis->local_fe_val_on_faces[i_face]->reinit(
      this->my_dealii_local_dofs_cell);
    bool i_am_on_sigma_s = BCs[i_face] & boundary_condition::semiconductor_R_I;
    bool i_am_on_sigma_e = BCs[i_face] & boundary_condition::electrolyte_R_I;
    bool i_am_on_sigma = i_am_on_sigma_e || i_am_on_sigma_s;
    if (i_am_on_sigma)
    {
      //
      // Loop 1
      for (unsigned i_face_quad = 0; i_face_quad < face_quad_size;
           ++i_face_quad)
      {
        double face_JxW = fe_face_val->JxW(i_face_quad);
        dealii::Tensor<1, dim> n_vec = fe_face_val->normal_vector(i_face_quad);
        (void)n_vec;
        //
        // We obtain rho_hat_n, rho_hat_p, rho_hat_r, rho_hat_o at the
        // quadrature point.
        //
        double rho_hat_n_at_quad = 0;
        double rho_hat_p_at_quad = 0;
        double rho_hat_r_at_quad = 0;
        double rho_hat_o_at_quad = 0;
        //
        for (unsigned j_face_unkn = 0; j_face_unkn < n_trace_unkns;
             ++j_face_unkn)
        {
          double lambda_j1 = fe_face_val->shape_value(j_face_unkn, i_face_quad);
          if (trace_unkns_is_active[0])
            rho_hat_n_at_quad += rho_n_hat[j_face_unkn] * lambda_j1;
          if (trace_unkns_is_active[1])
            rho_hat_p_at_quad += rho_p_hat[j_face_unkn] * lambda_j1;
          if (trace_unkns_is_active[2])
            rho_hat_r_at_quad += rho_r_hat[j_face_unkn] * lambda_j1;
          if (trace_unkns_is_active[3])
            rho_hat_o_at_quad += rho_o_hat[j_face_unkn] * lambda_j1;
        }
        //
        double Q1_at_quad = get_Q1(rho_hat_n_at_quad, rho_hat_o_at_quad);
        double Q2_at_quad = get_Q2(rho_hat_p_at_quad, rho_hat_r_at_quad);
        double d_Q1_d_n_at_quad = get_d_Q1_d_n(rho_hat_o_at_quad);
        double d_Q1_d_o_at_quad = get_d_Q1_d_o(rho_hat_n_at_quad);
        double d_Q2_d_p_at_quad = get_d_Q2_d_p(rho_hat_r_at_quad);
        double d_Q2_d_r_at_quad = get_d_Q2_d_r(rho_hat_p_at_quad);
        //
        if (trace_unkns_is_active[0] && trace_unkns_is_active[2])
        {
          for (unsigned i_face_unkn = 0; i_face_unkn < n_trace_unkns;
               ++i_face_unkn)
          {
            double lambda_i1 =
              fe_face_val->shape_value(i_face_unkn, i_face_quad);
            for (unsigned j_face_unkn = 0; j_face_unkn < n_trace_unkns;
                 ++j_face_unkn)
            {
              double lambda_j1 =
                fe_face_val->shape_value(j_face_unkn, i_face_quad);
              H11(i_face_unkn, j_face_unkn) +=
                lambda_i1 * lambda_j1 * face_JxW * d_Q1_d_n_at_quad;
              H14(i_face_unkn, j_face_unkn) +=
                lambda_i1 * lambda_j1 * face_JxW * d_Q1_d_o_at_quad;
              H22(i_face_unkn, j_face_unkn) +=
                lambda_i1 * lambda_j1 * face_JxW * d_Q2_d_p_at_quad;
              H23(i_face_unkn, j_face_unkn) +=
                lambda_i1 * lambda_j1 * face_JxW * d_Q2_d_r_at_quad;
            }
            Q1(i_face_unkn) += lambda_i1 * face_JxW * Q1_at_quad;
            Q2(i_face_unkn) += lambda_i1 * face_JxW * Q2_at_quad;
          }
        }
      }
      // Loop 1
    }
  }
  //
  if (this->local_equation_is_active[0])
  {
    Rn = 1. / mu_n * A1 * q_n_vec - B1 * rho_n_vec + C1 * rho_n_hat;
    Eigen::VectorXd Fn20, Fn21;
    Fn21 = (-B1.transpose() * q_n_vec - D1 * rho_n_vec + c_n * D2 * rho_n_vec +
            E1 * rho_n_hat - c_n * E2 * rho_n_hat) /
             2. -
           1. / dt * A1 * rho_n_vec;
    Fn20 = (-B1.transpose() * q_n_vec0 - D1 * rho_n_vec0 +
            c_n * D2 * rho_n_vec0 + E1 * rho_n_hat0 - c_n * E2 * rho_n_hat0) /
             2. +
           1. / dt * A1 * rho_n_vec0;
    Fn += (Fn20 + Fn21) / 2.;
    Ln += -C1.transpose() * q_n_vec - E1.transpose() * rho_n_vec +
          H1 * rho_n_hat - c_n * H2 * rho_n_hat;
  }
  if (this->trace_unkns_is_active[0])
    Ln += Q1;
  if (this->local_equation_is_active[1])
  {
    Rp = 1. / mu_p * A1 * q_p_vec - B1 * rho_p_vec + C1 * rho_p_hat;
    Eigen::VectorXd Fp20, Fp21;
    Fp21 = (-B1.transpose() * q_p_vec - D1 * rho_p_vec + c_p * D2 * rho_p_vec +
            E1 * rho_p_hat - c_p * E2 * rho_p_hat) /
             2. -
           1. / dt * A1 * rho_p_vec;
    Fp20 = (-B1.transpose() * q_p_vec0 - D1 * rho_p_vec0 +
            c_p * D2 * rho_p_vec0 + E1 * rho_p_hat0 - c_p * E2 * rho_p_hat0) /
             2. +
           1. / dt * A1 * rho_p_vec0;
    Fp += (Fp20 + Fp21) / 2.;
    Lp += -C1.transpose() * q_p_vec - E1.transpose() * rho_p_vec +
          H1 * rho_p_hat - c_p * H2 * rho_p_hat;
  }
  if (this->trace_unkns_is_active[1])
    Lp += Q2;
  if (this->local_equation_is_active[2])
  {
    Rr = 1. / mu_r * A1 * q_r_vec - B1 * rho_r_vec + C1 * rho_r_hat;
    Eigen::VectorXd Fr20, Fr21;
    Fr21 = (-B1.transpose() * q_r_vec - D1 * rho_r_vec + c_r * D2 * rho_r_vec +
            E1 * rho_r_hat - c_r * E2 * rho_r_hat) /
             2. -
           1. / dt * A1 * rho_r_vec;
    Fr20 = (-B1.transpose() * q_r_vec0 - D1 * rho_r_vec0 +
            c_r * D2 * rho_r_vec0 + E1 * rho_r_hat0 - c_r * E2 * rho_r_hat0) /
             2. +
           1. / dt * A1 * rho_r_vec0;
    Fr += (Fr20 + Fr21) / 2.;
    Lr += -C1.transpose() * q_r_vec - E1.transpose() * rho_r_vec +
          H1 * rho_r_hat - c_r * H2 * rho_r_hat;
  }
  if (this->trace_unkns_is_active[2])
    Lr += -Q1 + Q2;
  if (this->local_equation_is_active[3])
  {
    Ro = 1. / mu_o * A1 * q_o_vec - B1 * rho_o_vec + C1 * rho_o_hat;
    Eigen::VectorXd Fo20, Fo21;
    Fo21 = (-B1.transpose() * q_o_vec - D1 * rho_o_vec + c_o * D2 * rho_o_vec +
            E1 * rho_o_hat - c_o * E2 * rho_o_hat) /
             2. -
           1. / dt * A1 * rho_o_vec;
    Fo20 = (-B1.transpose() * q_o_vec0 - D1 * rho_o_vec0 +
            c_n * D2 * rho_o_vec0 + E1 * rho_o_hat0 - c_o * E2 * rho_o_hat0) /
             2. +
           1. / dt * A1 * rho_o_vec0;
    Fo += (Fo20 + Fo21) / 2.;
    Lo += -C1.transpose() * q_o_vec - E1.transpose() * rho_o_vec +
          H1 * rho_o_hat - c_o * H2 * rho_o_hat;
  }
  if (this->trace_unkns_is_active[3])
    Lo += Q1 - Q2;
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::add_my_nonlinear_terms()
{
  if (my_time_integrator_type == time_integrator_type::TRPZ)
  {
    compute_my_nonlinear_trpz_matrices();
  }
  // If we solve the problem with no time integration.
  {
    compute_my_nonlinear_matrices();
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::add_my_recombination_source()
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  //
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  unsigned cell_quad_size = my_basis->get_cell_quad_size();
  my_basis->local_fe_val_in_cell->reinit(this->my_dealii_local_dofs_cell);
  dealii::FEValuesExtractors::Scalar scalar(0);
  //
  double rho_i = own_cell->my_data->intrinsic_rho();
  double tau_n = own_cell->my_data->tau_n();
  double tau_p = own_cell->my_data->tau_p();
  //
  if (local_equation_is_active[0] && local_equation_is_active[1])
  {
    for (unsigned i_quad = 0; i_quad < cell_quad_size; ++i_quad)
    {
      double rho_n_at_quad = 0;
      double rho_p_at_quad = 0;
      double JxW = my_basis->local_fe_val_in_cell->JxW(i_quad);
      for (unsigned i1 = 0; i1 < n_scalar_unkns; ++i1)
      {
        double u_i1 =
          (*my_basis->local_fe_val_in_cell)[scalar].value(i1, i_quad);
        rho_n_at_quad += rho_n_vec[i1] * u_i1;
        rho_p_at_quad += rho_p_vec[i1] * u_i1;
      }
      for (unsigned j1 = 0; j1 < n_scalar_unkns; ++j1)
      {
        double u_j1 =
          (*my_basis->local_fe_val_in_cell)[scalar].value(j1, i_quad);
        double R_n_p =
          ((rho_n_at_quad * rho_p_at_quad) - rho_i * rho_i) /
          (tau_n * (rho_n_at_quad + rho_i) + tau_p * (rho_p_at_quad + rho_i));
        // ***
        Fn[j1] -= R_n_p * JxW * u_j1;
        Fp[j1] -= R_n_p * JxW * u_j1;
      }
    }
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::set_my_dyna_terms()
{
  if (my_time_integrator_type == time_integrator_type::BDF1)
  {
    ode_solvers::BDF1_solver<Eigen::VectorXd> *own_time_integrator =
      static_cast<ode_solvers::BDF1_solver<Eigen::VectorXd> *>(
        my_time_integrator);
    (void)own_time_integrator;
  }
  if (my_time_integrator_type == time_integrator_type::TRPZ)
  {
    ode_solvers::trapezoidal_solver<Eigen::VectorXd> *own_time_integrator =
      static_cast<ode_solvers::trapezoidal_solver<Eigen::VectorXd> *>(
        my_time_integrator);
    (void)own_time_integrator;
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<BasisType>::
  set_my_time_integrator(
    ode_solvers::first_order_ode_solver *in_time_integrator,
    time_integrator_type in_time_integrator_type)
{
  my_time_integrator = in_time_integrator;
  my_time_integrator_type = in_time_integrator_type;
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
  if (this->local_equation_is_active[0])
  {
    exact_rho_n.resize(n_scalar_unkns);
    exact_q_n.resize(n_scalar_unkns * dim);
  }
  if (this->local_equation_is_active[1])
  {
    exact_rho_p.resize(n_scalar_unkns);
    exact_q_p.resize(n_scalar_unkns * dim);
  }
  if (this->local_equation_is_active[2])
  {
    exact_rho_r.resize(n_scalar_unkns);
    exact_q_r.resize(n_scalar_unkns * dim);
  }
  if (this->local_equation_is_active[3])
  {
    exact_rho_o.resize(n_scalar_unkns);
    exact_q_o.resize(n_scalar_unkns * dim);
  }
  //
  my_basis->local_fe_val_at_cell_supp->reinit(this->my_dealii_local_dofs_cell);
  //
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    dealii::Point<spacedim> q_point =
      my_basis->local_fe_val_at_cell_supp->quadrature_point(i_unkn);
    //
    if (this->local_equation_is_active[0])
    {
      exact_rho_n(i_unkn) = own_cell->my_data->exact_rho_n(q_point);
      dealii::Tensor<1, dim> q_val = own_cell->my_data->exact_q_n(q_point);
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        exact_q_n(i_dim * n_scalar_unkns + i_unkn) = q_val[i_dim];
    }
    if (this->local_equation_is_active[1])
    {
      exact_rho_p(i_unkn) = own_cell->my_data->exact_rho_p(q_point);
      dealii::Tensor<1, dim> q_val = own_cell->my_data->exact_q_p(q_point);
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        exact_q_p(i_dim * n_scalar_unkns + i_unkn) = q_val[i_dim];
    }
    if (this->local_equation_is_active[2])
    {
      exact_rho_r(i_unkn) = own_cell->my_data->exact_rho_r(q_point);
      dealii::Tensor<1, dim> q_val = own_cell->my_data->exact_q_r(q_point);
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        exact_q_r(i_dim * n_scalar_unkns + i_unkn) = q_val[i_dim];
    }
    if (this->local_equation_is_active[3])
    {
      exact_rho_o(i_unkn) = own_cell->my_data->exact_rho_o(q_point);
      dealii::Tensor<1, dim> q_val = own_cell->my_data->exact_q_o(q_point);
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        exact_q_o(i_dim * n_scalar_unkns + i_unkn) = q_val[i_dim];
    }
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
  if (this->local_equation_is_active[0])
  {
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
  }
  //
  if (this->local_equation_is_active[1])
  {
    unsigned offset1 = (dim + 1) * n_scalar_unkns;
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
  }
  //
  if (this->local_equation_is_active[2])
  {
    unsigned offset1 = 2 * (dim + 1) * n_scalar_unkns;
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
  }
  //
  if (this->local_equation_is_active[3])
  {
    unsigned offset1 = 3 * (dim + 1) * n_scalar_unkns;
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
  // Here we assign values to trace_unkns_is_active
  //
  unsigned n_dof_per_face = my_basis->get_n_dofs_per_face();
  trace_unkns_is_active.resize(n_dof_per_face, 0);
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    for (unsigned i_unkn = 0; i_unkn < n_dof_per_face; ++i_unkn)
      trace_unkns_is_active[i_unkn] = trace_unkns_is_active[i_unkn] ||
                                      this->dof_status_on_faces[i_face][i_unkn];
  //
  // This assertion is required to make sure that on every element that
  // local_equation_is_active, trace_unknown_is_active as well ! BUT, not
  // the other way around.
  //
  for (unsigned i_unkn = 0; i_unkn < n_dof_per_face; ++i_unkn)
    if (local_equation_is_active[i_unkn])
      assert(trace_unkns_is_active[i_unkn]);
  //
  // Setting the size of gD, gN and f vectors
  //
  if (trace_unkns_is_active[0])
  {
    gD_rho_n = Eigen::VectorXd::Zero(my_basis->n_trace_unkns_per_cell_dof());
    gN_rho_n.resize(my_basis->n_trace_unkns_per_cell_dof(), 0.);
  }
  if (trace_unkns_is_active[1])
  {
    gD_rho_p = Eigen::VectorXd::Zero(my_basis->n_trace_unkns_per_cell_dof());
    gN_rho_p.resize(my_basis->n_trace_unkns_per_cell_dof(), 0.);
  }
  if (trace_unkns_is_active[2])
  {
    gD_rho_r = Eigen::VectorXd::Zero(my_basis->n_trace_unkns_per_cell_dof());
    gN_rho_r.resize(my_basis->n_trace_unkns_per_cell_dof(), 0.);
  }
  if (trace_unkns_is_active[3])
  {
    gD_rho_o = Eigen::VectorXd::Zero(my_basis->n_trace_unkns_per_cell_dof());
    gN_rho_o.resize(my_basis->n_trace_unkns_per_cell_dof(), 0.);
  }
  if (this->local_equation_is_active[0])
    f_n_vec = Eigen::VectorXd::Zero(n_scalar_unkns);
  if (this->local_equation_is_active[1])
    f_p_vec = Eigen::VectorXd::Zero(n_scalar_unkns);
  if (this->local_equation_is_active[2])
    f_r_vec = Eigen::VectorXd::Zero(n_scalar_unkns);
  if (this->local_equation_is_active[3])
    f_o_vec = Eigen::VectorXd::Zero(n_scalar_unkns);
  //
  my_basis->local_fe_val_at_cell_supp->reinit(this->my_dealii_local_dofs_cell);
  //
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    dealii::Point<spacedim> q_point =
      my_basis->local_fe_val_at_cell_supp->quadrature_point(i_unkn);
    //
    if (this->local_equation_is_active[0])
      f_n_vec[i_unkn] = own_cell->my_data->rho_n_rhs_func(q_point);
    if (this->local_equation_is_active[1])
      f_p_vec[i_unkn] = own_cell->my_data->rho_p_rhs_func(q_point);
    if (this->local_equation_is_active[2])
      f_r_vec[i_unkn] = own_cell->my_data->rho_r_rhs_func(q_point);
    if (this->local_equation_is_active[3])
      f_o_vec[i_unkn] = own_cell->my_data->rho_o_rhs_func(q_point);
  }
  //
  unsigned n_unkns_per_face_dof = my_basis->n_trace_unkns_per_face_dof();
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    my_basis->trace_fe_face_val_at_supp->reinit(this->my_dealii_trace_dofs_cell,
                                                i_face);
    const std::vector<dealii::Tensor<1, dim> > n_vecs =
      my_basis->trace_fe_face_val_at_supp->get_all_normal_vectors();
    const std::vector<dealii::Point<spacedim> > &face_supp_locs =
      my_basis->trace_fe_face_val_at_supp->get_quadrature_points();
    unsigned idx1 = i_face * n_unkns_per_face_dof;
    for (unsigned i1 = 0; i1 < n_unkns_per_face_dof; ++i1)
    {
      if (trace_unkns_is_active[0])
      {
        if (this->BCs[i_face] & boundary_condition::essential_rho_n)
        {
          double gD_at_face_supp =
            own_cell->my_data->gD_rho_n(face_supp_locs[i1]);
          gD_rho_n(idx1 + i1) = gD_at_face_supp;
        }
        if (this->BCs[i_face] & boundary_condition::natural_rho_n ||
            this->BCs[i_face] & boundary_condition::semiconductor_R_I)
        {
          dealii::Tensor<1, dim> gN_at_face_supp =
            own_cell->my_data->gN_rho_n(face_supp_locs[i1]);
          gN_rho_n[idx1 + i1] = gN_at_face_supp * n_vecs[i1];
        }
        if (this->BCs[i_face] & boundary_condition::semiconductor_R_I)
        {
          double RI_rhs_n_at_quad =
            own_cell->my_data->rhs_of_RI_n(face_supp_locs[i1]);
          gN_rho_n[idx1 + i1] -= RI_rhs_n_at_quad;
        }
      }
      //
      if (trace_unkns_is_active[1])
      {
        if (this->BCs[i_face] & boundary_condition::essential_rho_p)
        {
          double gD_at_face_supp =
            own_cell->my_data->gD_rho_p(face_supp_locs[i1]);
          gD_rho_p(idx1 + i1) = gD_at_face_supp;
        }
        if (this->BCs[i_face] & boundary_condition::natural_rho_p ||
            this->BCs[i_face] & boundary_condition::semiconductor_R_I)
        {
          dealii::Tensor<1, dim> gN_at_face_supp =
            own_cell->my_data->gN_rho_p(face_supp_locs[i1]);
          gN_rho_p[idx1 + i1] = gN_at_face_supp * n_vecs[i1];
        }
        if (this->BCs[i_face] & boundary_condition::semiconductor_R_I)
        {
          double RI_rhs_p_at_quad =
            own_cell->my_data->rhs_of_RI_p(face_supp_locs[i1]);
          gN_rho_p[idx1 + i1] -= RI_rhs_p_at_quad;
        }
      }
      //
      if (trace_unkns_is_active[2])
      {
        if (this->BCs[i_face] & boundary_condition::essential_rho_r)
        {
          double gD_at_face_supp =
            own_cell->my_data->gD_rho_r(face_supp_locs[i1]);
          gD_rho_r(idx1 + i1) = gD_at_face_supp;
        }
        if (this->BCs[i_face] & boundary_condition::natural_rho_r ||
            this->BCs[i_face] & boundary_condition::electrolyte_R_I)
        {
          dealii::Tensor<1, dim> gN_at_face_supp =
            own_cell->my_data->gN_rho_r(face_supp_locs[i1]);
          gN_rho_r[idx1 + i1] = gN_at_face_supp * n_vecs[i1];
        }
        if (this->BCs[i_face] & boundary_condition::electrolyte_R_I)
        {
          double RI_rhs_r_at_quad =
            own_cell->my_data->rhs_of_RI_r(face_supp_locs[i1]);
          gN_rho_r[idx1 + i1] -= RI_rhs_r_at_quad;
        }
      }
      //
      if (trace_unkns_is_active[3])
      {
        if (this->BCs[i_face] & boundary_condition::essential_rho_o)
        {
          double gD_at_face_supp =
            own_cell->my_data->gD_rho_o(face_supp_locs[i1]);
          gD_rho_o(idx1 + i1) = gD_at_face_supp;
        }
        if (this->BCs[i_face] & boundary_condition::natural_rho_o ||
            this->BCs[i_face] & boundary_condition::electrolyte_R_I)
        {
          dealii::Tensor<1, dim> gN_at_face_supp =
            own_cell->my_data->gN_rho_o(face_supp_locs[i1]);
          gN_rho_o[idx1 + i1] = gN_at_face_supp * n_vecs[i1];
        }
        if (this->BCs[i_face] & boundary_condition::electrolyte_R_I)
        {
          double RI_rhs_o_at_quad =
            own_cell->my_data->rhs_of_RI_o(face_supp_locs[i1]);
          gN_rho_o[idx1 + i1] -= RI_rhs_o_at_quad;
        }
      }
    }
  }
}

//
// ***

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::set_my_init_vals()
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  //
  //
  //
  if (trace_unkns_is_active[0])
    rho_n_hat = rho_n_hat0 = gD_rho_n;
  if (trace_unkns_is_active[1])
    rho_p_hat = rho_p_hat0 = gD_rho_p;
  if (trace_unkns_is_active[2])
    rho_r_hat = rho_r_hat0 = gD_rho_r;
  if (trace_unkns_is_active[3])
    rho_o_hat = rho_o_hat0 = gD_rho_o;
  //
  // Setting the size of initial values of rho_n,p,r,o.
  //
  if (this->local_equation_is_active[0])
  {
    rho_n_vec = rho_n_vec0 = Eigen::VectorXd::Zero(n_scalar_unkns);
    q_n_vec = q_n_vec0 = Eigen::VectorXd::Zero(n_scalar_unkns * dim);
  }
  if (this->local_equation_is_active[1])
  {
    rho_p_vec = rho_p_vec0 = Eigen::VectorXd::Zero(n_scalar_unkns);
    q_p_vec = q_p_vec0 = Eigen::VectorXd::Zero(n_scalar_unkns * dim);
  }
  if (this->local_equation_is_active[2])
  {
    rho_r_vec = rho_r_vec0 = Eigen::VectorXd::Zero(n_scalar_unkns);
    q_r_vec = q_r_vec0 = Eigen::VectorXd::Zero(n_scalar_unkns * dim);
  }
  if (this->local_equation_is_active[3])
  {
    rho_o_vec = rho_o_vec0 = Eigen::VectorXd::Zero(n_scalar_unkns);
    q_o_vec = q_o_vec0 = Eigen::VectorXd::Zero(n_scalar_unkns * dim);
  }
  //
  // Now, we get rho_n,p,r,o from the corresponding functions.
  //
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    dealii::Point<spacedim> q_point =
      my_basis->local_fe_val_at_cell_supp->quadrature_point(i_unkn);
    //
    if (this->local_equation_is_active[0])
      rho_n_vec[i_unkn] = rho_n_vec0[i_unkn] =
        own_cell->my_data->rho_n_0(q_point);
    if (this->local_equation_is_active[1])
      rho_p_vec[i_unkn] = rho_p_vec0[i_unkn] =
        own_cell->my_data->rho_p_0(q_point);
    if (this->local_equation_is_active[2])
      rho_r_vec[i_unkn] = rho_r_vec0[i_unkn] =
        own_cell->my_data->rho_r_0(q_point);
    if (this->local_equation_is_active[3])
      rho_o_vec[i_unkn] = rho_o_vec0[i_unkn] =
        own_cell->my_data->rho_o_0(q_point);
  }
}

//
// ***

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::set_my_trace_init_vals(const double *in_vec)
{
  unsigned n_trace_unkns = my_basis->n_trace_unkns_per_face_dof();
  //
  // Extracting d_rho_n_hat
  //
  if (trace_unkns_is_active[0])
  {
    for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
    {
      for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
      {
        unsigned j_unkn = 0 * n_trace_unkns + i_unkn;
        if (this->unkns_id_in_this_rank[i_face][j_unkn] >= 0)
          rho_n_hat(i_face * n_trace_unkns + i_unkn) =
            in_vec[this->unkns_id_in_this_rank[i_face][j_unkn]];
      }
    }
  }
  //
  // Extracting d_rho_p_hat
  //
  if (trace_unkns_is_active[1])
  {
    for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
    {
      for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
      {
        unsigned j_unkn = 1 * n_trace_unkns + i_unkn;
        if (this->unkns_id_in_this_rank[i_face][j_unkn] >= 0)
          rho_p_hat(i_face * n_trace_unkns + i_unkn) =
            in_vec[this->unkns_id_in_this_rank[i_face][j_unkn]];
      }
    }
  }
  //
  // Extracting d_rho_r_hat
  //
  if (trace_unkns_is_active[2])
  {
    for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
    {
      for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
      {
        unsigned j_unkn = 2 * n_trace_unkns + i_unkn;
        if (this->unkns_id_in_this_rank[i_face][j_unkn] >= 0)
          rho_r_hat(i_face * n_trace_unkns + i_unkn) =
            in_vec[this->unkns_id_in_this_rank[i_face][j_unkn]];
      }
    }
  }
  //
  // Extracting d_rho_o_hat
  //
  if (trace_unkns_is_active[3])
  {
    for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
    {
      for (unsigned i_unkn = 0; i_unkn < n_trace_unkns; ++i_unkn)
      {
        unsigned j_unkn = 3 * n_trace_unkns + i_unkn;
        if (this->unkns_id_in_this_rank[i_face][j_unkn] >= 0)
          rho_o_hat(i_face * n_trace_unkns + i_unkn) =
            in_vec[this->unkns_id_in_this_rank[i_face][j_unkn]];
      }
    }
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::get_my_L2_norm_of_NR_deltas(std::vector<double> *sum_of_NR_deltas)
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
    double d_rho_n_at_quad = 0;
    double d_rho_p_at_quad = 0;
    double d_rho_r_at_quad = 0;
    double d_rho_o_at_quad = 0;
    for (unsigned i1 = 0; i1 < n_scalar_unkns; ++i1)
    {
      double shape_val_u =
        my_basis->local_fe_val_in_cell->shape_value(i1, i_quad);
      if (this->local_equation_is_active[0])
        d_rho_n_at_quad += d_rho_n[i1] * shape_val_u;
      if (this->local_equation_is_active[1])
        d_rho_p_at_quad += d_rho_p[i1] * shape_val_u;
      if (this->local_equation_is_active[2])
        d_rho_r_at_quad += d_rho_r[i1] * shape_val_u;
      if (this->local_equation_is_active[3])
        d_rho_o_at_quad += d_rho_r[i1] * shape_val_u;
    }
    (*sum_of_NR_deltas)[0] += JxW * d_rho_n_at_quad * d_rho_n_at_quad;
    (*sum_of_NR_deltas)[1] += JxW * d_rho_p_at_quad * d_rho_p_at_quad;
    (*sum_of_NR_deltas)[2] += JxW * d_rho_r_at_quad * d_rho_r_at_quad;
    (*sum_of_NR_deltas)[3] += JxW * d_rho_o_at_quad * d_rho_o_at_quad;
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
      if (this->local_equation_is_active[0])
        rho_n_error_at_quad += (exact_rho_n(i1) - rho_n_vec(i1)) * shape_val_u;
      if (this->local_equation_is_active[1])
        rho_p_error_at_quad += (exact_rho_p(i1) - rho_p_vec(i1)) * shape_val_u;
      if (this->local_equation_is_active[2])
        rho_r_error_at_quad += (exact_rho_r(i1) - rho_r_vec(i1)) * shape_val_u;
      if (this->local_equation_is_active[3])
        rho_o_error_at_quad += (exact_rho_o(i1) - rho_o_vec(i1)) * shape_val_u;
      for (unsigned j1 = 0; j1 < dim; ++j1)
      {
        unsigned i2 = j1 * n_scalar_unkns + i1;
        if (this->local_equation_is_active[0])
          q_n_error_at_quad[j1] += shape_val_u * (exact_q_n(i2) - q_n_vec(i2));
        if (this->local_equation_is_active[1])
          q_p_error_at_quad[j1] += shape_val_u * (exact_q_p(i2) - q_p_vec(i2));
        if (this->local_equation_is_active[2])
          q_r_error_at_quad[j1] += shape_val_u * (exact_q_r(i2) - q_r_vec(i2));
        if (this->local_equation_is_active[3])
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
double
nargil::reactive_interface<dim, spacedim>::hdg_manager<BasisType>::get_Q1(
  const double in_rho_n, const double in_rho_o)
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  double k_et = own_cell->my_data->k_et();
  double rho_n_e = own_cell->my_data->rho_n_e();
  //
  return k_et * (in_rho_n - rho_n_e) * in_rho_o;
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
double
nargil::reactive_interface<dim, spacedim>::hdg_manager<BasisType>::get_Q2(
  const double in_rho_p, const double in_rho_r)
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  double k_ht = own_cell->my_data->k_ht();
  double rho_p_e = own_cell->my_data->rho_p_e();
  //
  return k_ht * (in_rho_p - rho_p_e) * in_rho_r;
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
double
nargil::reactive_interface<dim, spacedim>::hdg_manager<BasisType>::get_d_Q1_d_n(
  const double in_rho_o)
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  double k_et = own_cell->my_data->k_et();
  return k_et * in_rho_o;
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
double
nargil::reactive_interface<dim, spacedim>::hdg_manager<BasisType>::get_d_Q1_d_o(
  const double in_rho_n)
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  double k_et = own_cell->my_data->k_et();
  double rho_n_e = own_cell->my_data->rho_n_e();
  return k_et * (in_rho_n - rho_n_e);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
double
nargil::reactive_interface<dim, spacedim>::hdg_manager<BasisType>::get_d_Q2_d_p(
  const double in_rho_r)
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  double k_ht = own_cell->my_data->k_ht();
  return k_ht * in_rho_r;
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
double
nargil::reactive_interface<dim, spacedim>::hdg_manager<BasisType>::get_d_Q2_d_r(
  const double in_rho_p)
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  double k_ht = own_cell->my_data->k_ht();
  double rho_p_e = own_cell->my_data->rho_p_e();
  return k_ht * (in_rho_p - rho_p_e);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
bool nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::get_semiconductor_densities(Eigen::VectorXd &out_rho_n,
                                          Eigen::VectorXd &out_rho_p)
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  out_rho_n = own_cell->my_data->alpha_n() * rho_n_vec;
  out_rho_p = own_cell->my_data->alpha_p() * rho_p_vec;
  return (local_equation_is_active[0] && local_equation_is_active[1]);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
bool nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::get_electrolyte_densities(Eigen::VectorXd &out_rho_r,
                                        Eigen::VectorXd &out_rho_o)
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  out_rho_r = own_cell->my_data->alpha_r() * rho_r_vec;
  out_rho_o = own_cell->my_data->alpha_o() * rho_o_vec;
  return (local_equation_is_active[2] && local_equation_is_active[3]);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
template <typename RelevantCellManagerType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::get_my_E_from_relevant_cell()
{
  const reactive_interface *own_cell =
    static_cast<const reactive_interface *>(this->my_cell);
  diffusion<dim, spacedim> *other_cell = own_cell->my_relevant_diff_cell;
  RelevantCellManagerType *other_manager =
    other_cell->template get_manager<RelevantCellManagerType>();
  other_manager->set_flux_vector(&E_field, &E_star_dot_n);
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
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::set_init_vals(reactive_interface *in_cell)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->set_my_init_vals();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::set_trace_init_vals(reactive_interface *in_cell,
                                  const double *in_vec)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->set_my_trace_init_vals(in_vec);
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
  BasisType>::compute_linear_matrices(reactive_interface *in_cell)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->compute_my_linear_matrices();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::add_nonlinear_terms(reactive_interface *in_cell)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->add_my_nonlinear_terms();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::add_recombination_source(reactive_interface *in_cell)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->add_my_recombination_source();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::set_dyna_terms(reactive_interface *in_cell)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->set_my_dyna_terms();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<BasisType>::
  set_time_integrator(reactive_interface *in_cell,
                      ode_solvers::first_order_ode_solver *in_time_integrator,
                      time_integrator_type in_time_integrator_type)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->set_my_time_integrator(in_time_integrator,
                                      in_time_integrator_type);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::compute_NR_increments(reactive_interface *in_cell)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->compute_my_NR_increments();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::extract_NR_increment(reactive_interface *in_cell,
                                   const double *trace_sol)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->extract_my_NR_increment(trace_sol);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::get_L2_norm_of_NR_deltas(reactive_interface *in_cell,
                                       std::vector<double> *sum_of_NR_deltas)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->get_my_L2_norm_of_NR_deltas(sum_of_NR_deltas);
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
template <typename RelevantCellManagerType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::get_E_from_relevant_cell(reactive_interface *in_cell)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->template get_my_E_from_relevant_cell<RelevantCellManagerType>();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::reactive_interface<dim, spacedim>::hdg_manager<
  BasisType>::visualize_results(const viz_data &in_viz_data)
{
  unsigned time_level = in_viz_data.time_step;
  unsigned cycle = in_viz_data.cycle;
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
     dealii::Utilities::int_to_string(time_level, 4) +
     dealii::Utilities::int_to_string(cycle, 2));
  //
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);
  //
  if (comm_rank == 0)
  {
    std::vector<std::string> filenames;
    for (int i = 0; i < comm_size; ++i)
      filenames.push_back(in_viz_data.my_out_filename + "-" +
                          dealii::Utilities::int_to_string(i, 4) + "-" +
                          dealii::Utilities::int_to_string(time_level, 4) +
                          dealii::Utilities::int_to_string(cycle, 2) + ".vtu");
    std::ofstream master_output((filename + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}
