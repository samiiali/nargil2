#include "../../include/elements/diffusion.hpp"

//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::diffusion(
  dealiiTriCell<dim, spacedim> *inp_cell,
  const unsigned in_id_num,
  base_basis<dim, spacedim> *in_basis)
  : cell<dim, spacedim>(inp_cell, in_id_num), my_basis(in_basis)
{
}

//
//

template <int dim, int spacedim> nargil::diffusion<dim, spacedim>::~diffusion()
{
}

//
//

template <int dim, int spacedim>
template <typename CellManagerType, typename BasisType>
void nargil::diffusion<dim, spacedim>::diffusion::init_manager(
  const BasisType *in_basis)
{
  my_manager = std::move(
    std::unique_ptr<CellManagerType>(new CellManagerType(this, in_basis)));
}

//
//

template <int dim, int spacedim>
void nargil::diffusion<dim, spacedim>::assign_data(diffusion *in_cell,
                                                   data *in_data)
{
  in_cell->my_data = in_data;
}

//
//

template <int dim, int spacedim>
template <typename CellManagerType>
CellManagerType *nargil::diffusion<dim, spacedim>::get_manager()
{
  return static_cast<CellManagerType *>(my_manager.get());
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
const BasisType *nargil::diffusion<dim, spacedim>::get_basis() const
{
  return static_cast<const BasisType *>(my_basis);
}

//
//

template <int dim, int spacedim>
template <typename OtherCellEq>
void nargil::diffusion<dim, spacedim>::connect_to_other_cell(OtherCellEq *)
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
void nargil::diffusion<2, 2>::connect_to_other_cell(
  reactive_interface<2, 2> *in_relevant_cell)
{
  my_relvevant_R_I_cell = in_relevant_cell;
}
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::viz_data::viz_data(
  const MPI_Comm in_comm,
  const dealii::DoFHandler<dim, spacedim> *in_dof_handler,
  const dealii::LinearAlgebraPETSc::MPI::Vector *in_viz_sol,
  const std::string &in_filename, const std::string &in_u_name,
  const std::string &in_q_name)
  : my_comm(in_comm),
    my_dof_handler(in_dof_handler),
    my_viz_sol(in_viz_sol),
    my_out_filename(in_filename),
    my_u_name(in_u_name),
    my_q_name(in_q_name)
{
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::hdg_polybasis::hdg_polybasis(
  const unsigned poly_order, const unsigned quad_order)
  : base_basis<dim, spacedim>(),
    local_fe(dealii::FE_DGQ<dim>(poly_order), 1,
             dealii::FE_DGQ<dim>(poly_order), dim),
    trace_fe(poly_order),
    refn_fe(poly_order),
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
nargil::diffusion<dim, spacedim>::hdg_polybasis::~hdg_polybasis()
{
}

//
//

template <int dim, int spacedim>
dealii::Point<dim - 1>
nargil::diffusion<dim, spacedim>::hdg_polybasis::adjusted_subface_quad_points(
  const dealii::Point<dim - 1> &in_point, const unsigned half_range)
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
unsigned nargil::diffusion<dim, spacedim>::hdg_polybasis::get_n_dofs_per_face()
{
  return 1;
}

//
//

template <int dim, int spacedim>
std::vector<unsigned>
nargil::diffusion<dim, spacedim>::hdg_polybasis::get_n_unkns_per_dofs() const
{
  // Here, we will have copy elision, DO NOT try to optimize using move
  // semantics.
  std::vector<unsigned> n_unkns_per_dofs(get_n_dofs_per_face());
  n_unkns_per_dofs[0] = trace_fe.dofs_per_face;
  return n_unkns_per_dofs;
}

//
//

template <int dim, int spacedim>
const dealii::FESystem<dim> *
nargil::diffusion<dim, spacedim>::hdg_polybasis::get_local_fe() const
{
  return &local_fe;
}

//
//

template <int dim, int spacedim>
const dealii::FE_FaceQ<dim> *
nargil::diffusion<dim, spacedim>::hdg_polybasis::get_trace_fe() const
{
  return &trace_fe;
}

//
//

template <int dim, int spacedim>
const dealii::FE_DGQ<dim> *
nargil::diffusion<dim, spacedim>::hdg_polybasis::get_refn_fe() const
{
  return &refn_fe;
}

//
//

template <int dim, int spacedim>
const dealii::FESystem<dim> *
nargil::diffusion<dim, spacedim>::hdg_polybasis::get_viz_fe() const
{
  return &local_fe;
}

//
//

template <int dim, int spacedim>
unsigned
nargil::diffusion<dim, spacedim>::hdg_polybasis::get_face_quad_size() const
{
  return trace_fe_face_val[0]->get_quadrature().size();
}

//
//

template <int dim, int spacedim>
unsigned
nargil::diffusion<dim, spacedim>::hdg_polybasis::get_cell_quad_size() const
{
  return local_fe_val_in_cell->get_quadrature().size();
}

//
//

template <int dim, int spacedim>
unsigned
nargil::diffusion<dim, spacedim>::hdg_polybasis::n_unkns_per_local_scalar_dof()
  const
{
  return local_fe.base_element(0).dofs_per_cell;
}

//
//

template <int dim, int spacedim>
unsigned
nargil::diffusion<dim, spacedim>::hdg_polybasis::n_trace_unkns_per_cell() const
{
  return trace_fe.dofs_per_cell;
}

//
//

template <int dim, int spacedim>
unsigned
nargil::diffusion<dim, spacedim>::hdg_polybasis::n_trace_unkns_per_face() const
{
  return trace_fe.dofs_per_face;
}

//
//

template <int dim, int spacedim>
unsigned
nargil::diffusion<dim, spacedim>::hdg_polybasis::n_local_unkns_per_cell() const
{
  return local_fe.dofs_per_cell;
}

//
//
//
//
//

template <int dim, int spacedim>
template <typename BasisType>
nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::hdg_manager(
  const nargil::diffusion<dim, spacedim> *in_cell, const BasisType *in_basis)
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
nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::~hdg_manager()
{
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::assign_my_BCs(
  BC_Func f)
{
  f(this);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<
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
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::
  assemble_my_globals(solvers::base_implicit_solver<dim, spacedim> *in_solver)
{
  compute_my_matrices();
  //
  Eigen::MatrixXd A_inv = A.inverse();
  Eigen::LDLT<Eigen::MatrixXd> lu_of_Mat1(B.transpose() * A_inv * B + D);
  Eigen::MatrixXd Mat2 = (B.transpose() * A_inv * C + E);
  //
  std::vector<int> dof_indices(my_basis->n_trace_unkns_per_cell());
  for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
    for (unsigned i_unkn = 0; i_unkn < my_basis->n_trace_unkns_per_face();
         ++i_unkn)
    {
      int idx1 = i_face * my_basis->n_trace_unkns_per_face() + i_unkn;
      dof_indices[idx1] = this->unkns_id_in_all_ranks[i_face][i_unkn];
    }
  //
  for (unsigned i_dof = 0; i_dof < dof_indices.size(); ++i_dof)
  {
    u_vec = lu_of_Mat1.solve(Mat2.col(i_dof));
    q_vec = A_inv * (B * u_vec - C.col(i_dof));
    Eigen::VectorXd jth_col =
      C.transpose() * q_vec + E.transpose() * u_vec - H.col(i_dof);
    int i_col = dof_indices[i_dof];
    in_solver->push_to_global_mat(dof_indices.data(), &i_col, jth_col,
                                  ADD_VALUES);
  }
  //
  u_vec = lu_of_Mat1.solve(F + B.transpose() * A_inv * R + E * gD_vec);
  q_vec = A_inv * (-R + B * u_vec);
  Eigen::MatrixXd jth_col =
    L - C.transpose() * q_vec - E.transpose() * u_vec + H * gD_vec;
  in_solver->push_to_rhs_vec(dof_indices.data(), jth_col, ADD_VALUES);
  //
  // Now, we clean memory from the above matrices.
  //
  // this->free_up_memory(&A, &B, &C, &D, &E, &F, &H, &L, &R, &H0);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<
  BasisType>::compute_my_local_unkns(const double *trace_sol)
{
  // compute_my_matrices();
  //
  Eigen::MatrixXd A_inv = A.inverse();
  Eigen::LDLT<Eigen::MatrixXd> lu_of_Mat1(B.transpose() * A_inv * B + D);
  // This line requires that boundary condition files are projected to gD_vec
  uhat_vec = gD_vec;
  for (unsigned i_face = 0; i_face < this->my_cell->n_faces; ++i_face)
  {
    unsigned n_unkns = this->unkns_id_in_this_rank[i_face].size();
    for (unsigned i_unkn = 0; i_unkn < n_unkns; ++i_unkn)
      if (this->unkns_id_in_this_rank[i_face][i_unkn] >= 0)
        uhat_vec(i_face * n_unkns + i_unkn) =
          trace_sol[this->unkns_id_in_this_rank[i_face][i_unkn]];
  }
  u_vec = lu_of_Mat1.solve(F + (B.transpose() * A_inv * C + E) * uhat_vec);
  q_vec = A_inv * (B * u_vec - C * uhat_vec);
  grad_u_vec = -A0.inverse() * A * q_vec;
  //
  // *** This part can be moved to another function.
  //
  Eigen::LDLT<Eigen::MatrixXd> ldlt_of_H0(H0);
  q_star_dot_n_vec = ldlt_of_H0.solve(C.transpose() * q_vec +
                                      E.transpose() * u_vec - H * uhat_vec);
  //
  // Now we free the memory.
  //
  // this->free_up_memory(&A, &B, &C, &D, &E, &F, &H, &L, &R, &H0);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim,
                       spacedim>::hdg_manager<BasisType>::compute_my_matrices()
{
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  unsigned n_trace_unkns = my_basis->n_trace_unkns_per_cell();
  unsigned cell_quad_size = my_basis->get_cell_quad_size();
  unsigned face_quad_size = my_basis->get_face_quad_size();
  const std::vector<dealii::Point<spacedim> > &cell_quad_locs =
    my_basis->local_fe_val_in_cell->get_quadrature_points();
  const diffusion *own_cell = static_cast<const diffusion *>(this->my_cell);
  //
  dealii::FEValuesExtractors::Scalar scalar(0);
  dealii::FEValuesExtractors::Vector fluxes(1);
  //
  my_basis->local_fe_val_in_cell->reinit(this->my_dealii_local_dofs_cell);
  //
  A = Eigen::MatrixXd::Zero(dim * n_scalar_unkns, dim * n_scalar_unkns);
  B = Eigen::MatrixXd::Zero(dim * n_scalar_unkns, n_scalar_unkns);
  C = Eigen::MatrixXd::Zero(dim * n_scalar_unkns, n_trace_unkns);
  D = Eigen::MatrixXd::Zero(n_scalar_unkns, n_scalar_unkns);
  E = Eigen::MatrixXd::Zero(n_scalar_unkns, n_trace_unkns);
  F = Eigen::VectorXd::Zero(n_scalar_unkns);
  H = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  L = Eigen::VectorXd::Zero(n_trace_unkns);
  R = Eigen::VectorXd::Zero(n_trace_unkns);
  H0 = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  //
  // A0 matrix is the same as A, only does not have kappa_inv
  //
  A0 = Eigen::MatrixXd::Zero(dim * n_scalar_unkns, dim * n_scalar_unkns);
  //
  for (unsigned i_quad = 0; i_quad < cell_quad_size; ++i_quad)
  {
    double JxW = my_basis->local_fe_val_in_cell->JxW(i_quad);
    const dealii::Tensor<2, dim> &kappa_inv_at_quad =
      own_cell->my_data->kappa_inv(cell_quad_locs[i_quad]);
    for (unsigned i1 = n_scalar_unkns; i1 < (dim + 1) * n_scalar_unkns; ++i1)
    {
      dealii::Tensor<1, dim> q_i1 =
        (*my_basis->local_fe_val_in_cell)[fluxes].value(i1, i_quad);
      for (unsigned j1 = n_scalar_unkns; j1 < (dim + 1) * n_scalar_unkns; ++j1)
      {
        dealii::Tensor<1, dim> v_j1 =
          (*my_basis->local_fe_val_in_cell)[fluxes].value(j1, i_quad);
        A(i1 - n_scalar_unkns, j1 - n_scalar_unkns) +=
          kappa_inv_at_quad * q_i1 * v_j1 * JxW;
        A0(i1 - n_scalar_unkns, j1 - n_scalar_unkns) += q_i1 * v_j1 * JxW;
      }
    }
    //
    for (unsigned j1 = 0; j1 < n_scalar_unkns; ++j1)
    {
      double u_j1 = (*my_basis->local_fe_val_in_cell)[scalar].value(j1, i_quad);
      for (unsigned i1 = 0; i1 < n_scalar_unkns; ++i1)
      {
        double u_i1 =
          (*my_basis->local_fe_val_in_cell)[scalar].value(i1, i_quad);
        double mij = u_i1 * u_j1 * JxW;
        F(j1) += mij * f_vec(i1);
      }
      for (unsigned i1 = n_scalar_unkns; i1 < (dim + 1) * n_scalar_unkns; ++i1)
      {
        double q_i1_div =
          (*my_basis->local_fe_val_in_cell)[fluxes].divergence(i1, i_quad);
        B(i1 - n_scalar_unkns, j1) += u_j1 * q_i1_div * JxW;
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
      dealii::Tensor<1, dim> n_vec = fe_face_val->normal_vector(i_face_quad);
      double face_JxW = fe_face_val->JxW(i_face_quad);
      //
      const dealii::Tensor<2, dim> &kappa_inv_at_quad =
        own_cell->my_data->kappa_inv(face_quad_locs[i_face_quad]);
      const dealii::Tensor<2, dim> &kappa_at_quad =
        dealii::invert(kappa_inv_at_quad);
      double tau_at_quad = own_cell->my_data->tau(face_quad_locs[i_face_quad]);
      tau_at_quad *= n_vec * kappa_at_quad * n_vec;
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
          D(i1, j1) += face_JxW * tau_at_quad * u_i1 * w_j1;
        }
      }
      //
      // Here we compute the value of gN at quadrature point.
      //
      //
      // double to_be_1 = 0;
      //
      dealii::Tensor<1, dim> gN_at_face_quad;
      for (unsigned j_face_unkn = 0; j_face_unkn < n_trace_unkns; ++j_face_unkn)
      {
        double lambda_j1 = fe_face_val->shape_value(j_face_unkn, i_face_quad);
        gN_at_face_quad += gN_vec[j_face_unkn] * lambda_j1;
      }
      double gN_dot_n = gN_at_face_quad * n_vec;
      // Loop 2
      for (unsigned i_face_unkn = 0; i_face_unkn < n_trace_unkns; ++i_face_unkn)
      {
        double lambda_i1 = fe_face_val->shape_value(i_face_unkn, i_face_quad);
        //
        for (unsigned j1 = n_scalar_unkns; j1 < (dim + 1) * n_scalar_unkns;
             ++j1)
        {
          dealii::Tensor<1, dim> v_j1 =
            (*my_basis->local_fe_val_on_faces[i_face])[fluxes].value(
              j1, i_face_quad);
          C(j1 - n_scalar_unkns, i_face_unkn) +=
            face_JxW * lambda_i1 * (v_j1 * n_vec);
        }
        //
        for (unsigned j1 = 0; j1 < n_scalar_unkns; ++j1)
        {
          double w_j1 =
            (*my_basis->local_fe_val_on_faces[i_face])[scalar].value(
              j1, i_face_quad);
          E(j1, i_face_unkn) += w_j1 * tau_at_quad * lambda_i1 * face_JxW;
        }
        //
        for (unsigned j_face_unkn = 0; j_face_unkn < n_trace_unkns;
             ++j_face_unkn)
        {
          double lambda_j1 = fe_face_val->shape_value(j_face_unkn, i_face_quad);
          H(i_face_unkn, j_face_unkn) +=
            lambda_i1 * tau_at_quad * lambda_j1 * face_JxW;
          //
          // *** This part can be put into another function.
          //
          H0(i_face_unkn, j_face_unkn) += lambda_i1 * lambda_j1 * face_JxW;
        }
        L(i_face_unkn) += lambda_i1 * face_JxW * gN_dot_n;
      }
      // Loop 2
    }
    // Loop 1
  }
  R = C * gD_vec;
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<
  BasisType>::interpolate_to_my_trace()
{
  const diffusion *own_cell = static_cast<const diffusion *>(this->my_cell);
  exact_uhat.resize(my_basis->trace_fe.n_dofs_per_cell());
  unsigned num1 = 0;
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    my_basis->trace_fe_face_val_at_supp->reinit(this->my_dealii_trace_dofs_cell,
                                                i_face);
    const std::vector<dealii::Point<spacedim> > &face_supp_locs =
      my_basis->trace_fe_face_val_at_supp->get_quadrature_points();
    for (unsigned i1 = 0; i1 < my_basis->n_trace_unkns_per_face(); ++i1)
    {
      exact_uhat(num1) = own_cell->my_data->exact_u(face_supp_locs[i1]);
      ++num1;
    }
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<
  BasisType>::interpolate_to_my_interior()
{
  const diffusion *own_cell = static_cast<const diffusion *>(this->my_cell);
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  exact_u.resize(n_scalar_unkns);
  exact_q.resize(n_scalar_unkns * dim);
  //
  my_basis->local_fe_val_at_cell_supp->reinit(this->my_dealii_local_dofs_cell);
  //
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    dealii::Point<spacedim> q_point =
      my_basis->local_fe_val_at_cell_supp->quadrature_point(i_unkn);
    exact_u(i_unkn) = own_cell->my_data->exact_u(q_point);
    dealii::Tensor<1, dim> q_val = own_cell->my_data->exact_q(q_point);
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
    {
      exact_q(i_dim * n_scalar_unkns + i_unkn) = q_val[i_dim];
    }
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<
  BasisType>::fill_my_viz_vector(distributed_vector<dim, spacedim> *out_vec)
{
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  //
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    int idx1 = this->local_interior_unkn_idx[i_unkn];
    out_vec->assemble(idx1, u_vec(i_unkn));
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
    {
      int idx2 =
        this->local_interior_unkn_idx[(i_dim + 1) * n_scalar_unkns + i_unkn];
      out_vec->assemble(idx2, q_vec(i_dim * n_scalar_unkns + i_unkn));
    }
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::
  fill_my_viz_vector_with_grad_u_dot_b(
    distributed_vector<dim, spacedim> *out_vec,
    std::function<dealii::Tensor<1, dim>(const dealii::Point<spacedim> &p)>
      b_func)
{
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  //
  my_basis->local_fe_val_at_cell_supp->reinit(this->my_dealii_local_dofs_cell);
  //
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    dealii::Point<spacedim> q_point =
      my_basis->local_fe_val_at_cell_supp->quadrature_point(i_unkn);
    dealii::Tensor<1, dim> b_comps_at_q_point = b_func(q_point);
    dealii::Tensor<1, dim> grad_u_at_q_point;
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
    {
      int idx2 =
        this->local_interior_unkn_idx[(i_dim + 1) * n_scalar_unkns + i_unkn];
      out_vec->assemble(idx2, grad_u_vec(i_dim * n_scalar_unkns + i_unkn));
      grad_u_at_q_point[i_dim] = grad_u_vec(i_dim * n_scalar_unkns + i_unkn);
    }
    double b_dot_grad_u_at_q_point = grad_u_at_q_point * b_comps_at_q_point;
    int idx1 = this->local_interior_unkn_idx[i_unkn];
    out_vec->assemble(idx1, b_dot_grad_u_at_q_point);
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::
  fill_my_viz_vec_with_exact_sol(distributed_vector<dim, spacedim> *out_vec)
{
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  //
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    int idx1 = this->local_interior_unkn_idx[i_unkn];
    out_vec->assemble(idx1, exact_u(i_unkn));
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
    {
      int idx2 =
        this->local_interior_unkn_idx[(i_dim + 1) * n_scalar_unkns + i_unkn];
      out_vec->assemble(idx2, exact_q(i_dim * n_scalar_unkns + i_unkn));
    }
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<
  BasisType>::fill_my_refn_vector(distributed_vector<dim, spacedim> *out_vec)
{
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  //
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    int idx1 = this->my_cell->id_num * n_scalar_unkns + i_unkn;
    out_vec->assemble(idx1, u_vec(i_unkn));
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<
  BasisType>::set_my_source_and_BCs()
{
  const diffusion *own_cell = static_cast<const diffusion *>(this->my_cell);
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  f_vec = Eigen::VectorXd::Zero(n_scalar_unkns);
  //
  my_basis->local_fe_val_at_cell_supp->reinit(this->my_dealii_local_dofs_cell);
  //
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    dealii::Point<spacedim> q_point =
      my_basis->local_fe_val_at_cell_supp->quadrature_point(i_unkn);
    double value_at_i_node = own_cell->my_data->rhs_func(q_point);
    f_vec(i_unkn) = value_at_i_node;
  }
  //
  gD_vec = Eigen::VectorXd::Zero(my_basis->n_trace_unkns_per_cell());
  gN_vec.resize(my_basis->n_trace_unkns_per_cell());
  unsigned n_dofs_per_face = my_basis->n_trace_unkns_per_face();
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    my_basis->trace_fe_face_val_at_supp->reinit(this->my_dealii_trace_dofs_cell,
                                                i_face);
    const std::vector<dealii::Point<spacedim> > &face_supp_locs =
      my_basis->trace_fe_face_val_at_supp->get_quadrature_points();
    for (unsigned i1 = 0; i1 < n_dofs_per_face; ++i1)
    {
      unsigned idx1 = i_face * n_dofs_per_face;
      if (this->BCs[i_face] == boundary_condition::essential)
      {
        double gD_at_face_supp = own_cell->my_data->gD_func(face_supp_locs[i1]);
        gD_vec(idx1 + i1) = gD_at_face_supp;
      }
      if (this->BCs[i_face] == boundary_condition::natural)
      {
        dealii::Tensor<1, dim> gN_at_face_supp =
          own_cell->my_data->gN_func(face_supp_locs[i1]);
        gN_vec[idx1 + i1] = gN_at_face_supp;
      }
    }
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
template <typename RelevantCellManagerType>
void nargil::diffusion<dim,
                       spacedim>::hdg_manager<BasisType>::apply_my_R_I_source()
{
  Eigen::VectorXd alpha_rho_n, alpha_rho_p, alpha_rho_r, alpha_rho_o;
  const diffusion<dim, spacedim> *own_cell =
    static_cast<const diffusion *>(this->my_cell);
  reactive_interface<dim, spacedim> *other_cell =
    own_cell->my_relvevant_R_I_cell;
  RelevantCellManagerType *other_manager =
    other_cell->template get_manager<RelevantCellManagerType>();
  bool in_semiconductor =
    other_manager->get_semiconductor_densities(alpha_rho_n, alpha_rho_p);
  bool in_electrolyte =
    other_manager->get_electrolyte_densities(alpha_rho_r, alpha_rho_o);
  //
  unsigned n_scalar_unkns = my_basis->n_unkns_per_local_scalar_dof();
  unsigned cell_quad_size = my_basis->get_cell_quad_size();
  my_basis->local_fe_val_in_cell->reinit(this->my_dealii_local_dofs_cell);
  dealii::FEValuesExtractors::Scalar scalar(0);
  //
  if (in_semiconductor)
  {
    for (unsigned i_quad = 0; i_quad < cell_quad_size; ++i_quad)
    {
      double alpha_rho_n_at_quad = 0;
      double alpha_rho_p_at_quad = 0;
      double JxW = my_basis->local_fe_val_in_cell->JxW(i_quad);
      for (unsigned i1 = 0; i1 < n_scalar_unkns; ++i1)
      {
        double u_i1 =
          (*my_basis->local_fe_val_in_cell)[scalar].value(i1, i_quad);
        alpha_rho_n_at_quad += alpha_rho_n[i1] * u_i1;
        alpha_rho_p_at_quad += alpha_rho_p[i1] * u_i1;
      }
      for (unsigned j1 = 0; j1 < n_scalar_unkns; ++j1)
      {
        double u_j1 =
          (*my_basis->local_fe_val_in_cell)[scalar].value(j1, i_quad);
        F[j1] += (alpha_rho_n_at_quad + alpha_rho_p_at_quad) * JxW * u_j1;
      }
    }
  }
  else if (in_electrolyte)
  {
    for (unsigned i_quad = 0; i_quad < cell_quad_size; ++i_quad)
    {
      double alpha_rho_r_at_quad = 0;
      double alpha_rho_o_at_quad = 0;
      double JxW = my_basis->local_fe_val_in_cell->JxW(i_quad);
      for (unsigned i1 = 0; i1 < n_scalar_unkns; ++i1)
      {
        double u_i1 =
          (*my_basis->local_fe_val_in_cell)[scalar].value(i1, i_quad);
        alpha_rho_r_at_quad += alpha_rho_r[i1] * u_i1;
        alpha_rho_o_at_quad += alpha_rho_o[i1] * u_i1;
      }
      for (unsigned j1 = 0; j1 < n_scalar_unkns; ++j1)
      {
        double u_j1 =
          (*my_basis->local_fe_val_in_cell)[scalar].value(j1, i_quad);
        F[j1] += (alpha_rho_r_at_quad + alpha_rho_o_at_quad) * JxW * u_j1;
      }
    }
  }
  else
    assert(false);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<
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
    for (unsigned i1 = 0; i1 < n_scalar_unkns; ++i1)
    {
      double u_error = exact_u(i1) - u_vec(i1);
      double shape_val_u =
        my_basis->local_fe_val_in_cell->shape_value(i1, i_quad);
      (*sum_of_L2_errors)[0] += shape_val_u * JxW * u_error * u_error;
    }
    for (unsigned i1 = n_scalar_unkns; i1 < (dim + 1) * n_scalar_unkns; ++i1)
    {
      unsigned i2 = i1 - n_scalar_unkns;
      double q_error = exact_q(i2) - q_vec(i2);
      double shape_val_q =
        my_basis->local_fe_val_in_cell->shape_value(i1, i_quad);
      (*sum_of_L2_errors)[1] += shape_val_q * JxW * q_error * q_error;
    }
  }
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::set_flux_vector(
  double **out_q, double **out_q_star)
{
  *out_q = q_vec.data();
  *out_q_star = q_star_dot_n_vec.data();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::assign_BCs(
  diffusion *in_cell, BC_Func func)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->assign_my_BCs(func);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<
  BasisType>::interpolate_to_trace(diffusion *in_cell)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->interpolate_to_my_trace();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<
  BasisType>::interpolate_to_interior(diffusion *in_cell)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->interpolate_to_my_interior();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::fill_viz_vector(
  diffusion *in_cell, distributed_vector<dim, spacedim> *out_vec)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->fill_my_viz_vector(out_vec);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::
  fill_viz_vector_with_grad_u_dot_b(
    diffusion *in_cell, distributed_vector<dim, spacedim> *out_vec,
    std::function<dealii::Tensor<1, dim>(const dealii::Point<spacedim> &p)>
      b_func)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->fill_my_viz_vector_with_grad_u_dot_b(out_vec, b_func);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::
  fill_viz_vec_with_exact_sol(diffusion *in_cell,
                              distributed_vector<dim, spacedim> *out_vec)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->fill_my_viz_vec_with_exact_sol(out_vec);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::fill_refn_vector(
  diffusion *in_cell, distributed_vector<dim, spacedim> *out_vec)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->fill_my_refn_vector(out_vec);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<
  BasisType>::set_source_and_BCs(diffusion *in_cell)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->set_my_source_and_BCs();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
template <typename RelevantCellManagerType>
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::apply_R_I_source(
  diffusion *in_cell)
{
  hdg_manager *own_manager = in_cell->template get_manager<hdg_manager>();
  own_manager->template apply_my_R_I_source<RelevantCellManagerType>();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::assemble_globals(
  diffusion *in_cell, solvers::base_implicit_solver<dim, spacedim> *in_solver)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->assemble_my_globals(in_solver);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::compute_matrices(
  diffusion *in_cell)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->compute_my_matrices();
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<
  BasisType>::compute_local_unkns(diffusion *in_cell, const double *trace_sol)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->compute_my_local_unkns(trace_sol);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<BasisType>::compute_errors(
  diffusion *in_cell, std::vector<double> *sum_of_L2_errors)
{
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(in_cell->my_manager.get());
  own_manager->compute_my_errors(sum_of_L2_errors);
}

//
//

template <int dim, int spacedim>
template <typename BasisType>
void nargil::diffusion<dim, spacedim>::hdg_manager<
  BasisType>::visualize_results(const viz_data &in_viz_data)
{
  unsigned time_level = in_viz_data.time_step;
  unsigned cycle = in_viz_data.cycle;
  const auto &tria = in_viz_data.my_dof_handler->get_triangulation();
  unsigned n_active_cells = tria.n_active_cells();
  int comm_rank, comm_size;
  MPI_Comm_rank(in_viz_data.my_comm, &comm_rank);
  MPI_Comm_size(in_viz_data.my_comm, &comm_size);
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(*in_viz_data.my_dof_handler);
  std::vector<std::string> solution_names(dim + 1);
  solution_names[0] = in_viz_data.my_u_name;
  for (unsigned i1 = 0; i1 < dim; ++i1)
    solution_names[i1 + 1] = in_viz_data.my_q_name;
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      1, dealii::DataComponentInterpretation::component_is_scalar);
  for (unsigned i1 = 0; i1 < dim; ++i1)
    data_component_interpretation.push_back(
      dealii::DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector(*in_viz_data.my_viz_sol,
                           solution_names,
                           dealii::DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  dealii::Vector<float> subdomain(n_active_cells);
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = comm_rank;
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  const std::string filename =
    (in_viz_data.my_out_filename + "-" +
     dealii::Utilities::int_to_string(comm_rank, 4) + "-" +
     dealii::Utilities::int_to_string(time_level, 4) +
     dealii::Utilities::int_to_string(cycle, 2));
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

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
