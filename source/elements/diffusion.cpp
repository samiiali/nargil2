#include "../../include/elements/diffusion.hpp"

//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::diffusion(dealiiTriCell *inp_cell,
                                            const unsigned in_id_num,
                                            base_basis<dim, spacedim> *in_basis,
                                            base_model *model_)
  : cell<dim, spacedim>(inp_cell, in_id_num, model_), my_basis(in_basis)
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
template <typename CellManagerType>
void nargil::diffusion<dim, spacedim>::diffusion::init_manager()
{
  my_manager =
    std::move(std::unique_ptr<CellManagerType>(new CellManagerType(this)));
}

//
//

template <int dim, int spacedim>
template <typename BasisType, typename Func>
void nargil::diffusion<dim, spacedim>::assign_BCs(Func f)
{
  static_cast<typename BasisType::CellManagerType *>(my_manager.get())
    ->assign_BCs(f);
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
//
//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::hdg_polybasis::hdg_polybasis(
  const unsigned poly_order, const unsigned quad_order)
  : base_basis<dim, spacedim>(),
    _poly_order(poly_order),
    _quad_order(quad_order),
    local_fe(dealii::FE_DGQ<dim>(poly_order), 1,
             dealii::FE_DGQ<dim>(poly_order), dim),
    trace_fe(poly_order),
    cell_quad(quad_order),
    face_quad(quad_order),
    local_fe_val_on_faces(2 * dim)
{
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
  std::unique_ptr<dealii::FEValues<dim> > local_fe_vel_temp(
    new dealii::FEValues<dim>(local_fe, cell_quad, fe_val_flags));
  //
  std::unique_ptr<dealii::FEValues<dim> > local_fe_vel_at_supp_temp(
    new dealii::FEValues<dim>(local_fe, local_fe.get_unit_support_points(),
                              fe_val_flags_at_supp));
  //
  std::unique_ptr<dealii::FEFaceValues<dim> > trace_fe_face_val_temp(
    new dealii::FEFaceValues<dim>(trace_fe, face_quad, fe_face_val_flags));

  local_fe_val_in_cell = std::move(local_fe_vel_temp);
  local_fe_val_at_cell_supp = std::move(local_fe_vel_at_supp_temp);
  trace_fe_face_val = std::move(trace_fe_face_val_temp);

  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    dealii::Quadrature<dim> projected_i_face_quad =
      dealii::QProjector<dim>::project_to_face(face_quad, i_face);
    std::unique_ptr<dealii::FEValues<dim> > local_fe_vel_on_i_face(
      new dealii::FEValues<dim>(local_fe, projected_i_face_quad, fe_val_flags));
    local_fe_val_on_faces[i_face] = std::move(local_fe_vel_on_i_face);
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
void nargil::diffusion<dim, spacedim>::hdg_polybasis::
  adjusted_subface_quad_points(const dealii::Point<dim - 1> &in_point,
                               const unsigned half_range)
{
  assert(half_range <= pow(2, in_point.dimension));
  std::vector<double> result;
  dealii::Point<dim - 1> out_point(in_point);
  if (half_range != 0)
  {
    if (in_point.dimension == 1)
    {
      if (half_range == 1)
        out_point(0) = in_point(0) / 2.0;
      if (half_range == 2)
        out_point(0) = 0.5 + in_point(0) / 2.0;
    }
    if (in_point.dimension == 2)
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
//
//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::hdg_manager::hdg_manager(
  const nargil::diffusion<dim, spacedim> *in_cell)
  : hybridized_cell_manager<dim, spacedim>(in_cell)
{
  const hdg_polybasis *my_basis = in_cell->template get_basis<hdg_polybasis>();
  local_interior_unkn_idx.resize(my_basis->local_fe.dofs_per_cell);
}

//
//

template <int dim, int spacedim>
nargil::diffusion<dim, spacedim>::hdg_manager::~hdg_manager()
{
}

//
//

template <int dim, int spacedim>
template <typename Func>
void nargil::diffusion<dim, spacedim>::hdg_manager::assign_BCs(Func f)
{
  f(this);
}

//
//

template <int dim, int spacedim>
void nargil::diffusion<dim, spacedim>::hdg_manager::
  assign_local_interior_unkn_id(unsigned *local_num)
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
void nargil::diffusion<dim, spacedim>::hdg_manager::assemble_globals()
{
  compute_matrices();
}

//
//

template <int dim, int spacedim>
void nargil::diffusion<dim, spacedim>::hdg_manager::compute_local_unkns()
{
  compute_matrices();
  Eigen::MatrixXd A_inv = A.inverse();
  // *** Compute u_out, instead of the following line *** //
  u_vec = exact_u;
  q_vec = A_inv * (R + B * u_vec - C * exact_uhat);
}

//
//

template <int dim, int spacedim>
void nargil::diffusion<dim, spacedim>::hdg_manager::compute_matrices()
{
  const diffusion *own_cell = static_cast<const diffusion *>(this->my_cell);
  hdg_polybasis *own_basis = static_cast<hdg_polybasis *>(own_cell->my_basis);
  unsigned n_scalar_unkns = own_basis->local_fe.base_element(0).dofs_per_cell;
  unsigned n_trace_unkns = own_basis->trace_fe.dofs_per_cell;
  //  unsigned n_trace_unkns = pow(my_own_basis->_poly_order + 1, dim);

  dealii::FEValuesExtractors::Scalar scalar(0);
  dealii::FEValuesExtractors::Vector fluxes(1);

  unsigned n_trace_unkns_per_face = own_basis->trace_fe.dofs_per_face;

  //  unsigned n_u_unkns = my_own_basis->u_basis.dofs_per_cell;

  own_basis->local_fe_val_in_cell->reinit(this->my_dealii_local_dofs_cell);

  A = Eigen::MatrixXd::Zero(dim * n_scalar_unkns, dim * n_scalar_unkns);
  B = Eigen::MatrixXd::Zero(dim * n_scalar_unkns, n_scalar_unkns);
  C = Eigen::MatrixXd::Zero(dim * n_scalar_unkns, n_trace_unkns);
  H = Eigen::MatrixXd::Zero(n_trace_unkns, n_trace_unkns);
  R = Eigen::VectorXd::Zero(n_trace_unkns);

  for (unsigned i_quad = 0; i_quad < own_basis->cell_quad.size(); ++i_quad)
  {
    double JxW = own_basis->local_fe_val_in_cell->JxW(i_quad);
    for (unsigned i1 = n_scalar_unkns; i1 < (dim + 1) * n_scalar_unkns; ++i1)
    {
      dealii::Tensor<1, dim> q_i1 =
        (*own_basis->local_fe_val_in_cell)[fluxes].value(i1, i_quad);
      for (unsigned j1 = n_scalar_unkns; j1 < (dim + 1) * n_scalar_unkns; ++j1)
      {
        dealii::Tensor<1, dim> v_j1 =
          (*own_basis->local_fe_val_in_cell)[fluxes].value(j1, i_quad);
        A(i1 - n_scalar_unkns, j1 - n_scalar_unkns) += q_i1 * v_j1 * JxW;
      }
    }

    for (unsigned i1 = n_scalar_unkns; i1 < (dim + 1) * n_scalar_unkns; ++i1)
    {
      double q_i1_div =
        (*own_basis->local_fe_val_in_cell)[fluxes].divergence(i1, i_quad);
      for (unsigned j1 = 0; j1 < n_scalar_unkns; ++j1)
      {
        double u_j1 =
          (*own_basis->local_fe_val_in_cell)[scalar].value(j1, i_quad);
        B(i1 - n_scalar_unkns, j1) += u_j1 * q_i1_div * JxW;
      }
    }
  }

  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    own_basis->trace_fe_face_val->reinit(this->my_dealii_trace_dofs_cell,
                                         i_face);
    for (unsigned i_face_quad = 0; i_face_quad < own_basis->face_quad.size();
         ++i_face_quad)
    {
      double face_JxW = own_basis->trace_fe_face_val->JxW(i_face_quad);
      dealii::Tensor<1, dim> n_vec =
        own_basis->trace_fe_face_val->normal_vector(i_face_quad);
      unsigned i_face_unkn = 0;
      for (unsigned i_unkn_per_face = 0;
           i_unkn_per_face < n_trace_unkns_per_face;
           ++i_unkn_per_face)
      {
        double lambda_i1 =
          own_basis->trace_fe_face_val->shape_value(i_face_unkn, i_face_quad);
        for (unsigned j1 = n_scalar_unkns; j1 < (dim + 1) * n_scalar_unkns;
             ++j1)
        {
          dealii::Tensor<1, dim> v_j1 =
            (*own_basis->local_fe_val_in_cell)[fluxes].value(j1, i_face_quad);
          C(j1 - n_scalar_unkns, i_face_unkn) +=
            face_JxW * lambda_i1 * (v_j1 * n_vec);
        }
        unsigned j_face_unkn = 0;
        for (unsigned i_unkn_per_face = 0;
             i_unkn_per_face < n_trace_unkns_per_face;
             ++i_unkn_per_face)
        {
          double lambda_j1 =
            own_basis->trace_fe_face_val->shape_value(j_face_unkn, i_face_quad);
          H(i_face_unkn, j_face_unkn) += lambda_i1 * lambda_j1 * face_JxW;
          ++j_face_unkn;
        }
        ++i_face_unkn;
      }
    }
  }

  R = C * exact_uhat;
}

//
//

template <int dim, int spacedim>
void nargil::diffusion<dim, spacedim>::hdg_manager::run_interpolate_to_trace(
  nargil::cell<dim, spacedim> *in_cell, funcType func)
{
  diffusion *own_cell = static_cast<diffusion *>(in_cell);
  const hdg_polybasis *own_basis =
    own_cell->template get_basis<hdg_polybasis>();
  hdg_manager *own_manager = own_cell->template get_manager<hdg_manager>();
  own_manager->exact_uhat.resize(own_basis->trace_fe.n_dofs_per_cell());
  unsigned num1 = 0;
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    std::vector<double> exact_uhat_vec_on_face(
      own_basis->trace_fe.dofs_per_face);
    own_basis->trace_fe_face_val->reinit(own_manager->my_dealii_trace_dofs_cell,
                                         i_face);
    const std::vector<dealii::Point<spacedim> > &face_quad_locs =
      own_basis->trace_fe_face_val->get_quadrature_points();
    std::transform(face_quad_locs.begin(), face_quad_locs.end(),
                   exact_uhat_vec_on_face.begin(), func);
    for (unsigned i1 = 0; i1 < own_basis->trace_fe.n_dofs_per_face(); ++i1)
    {
      own_manager->exact_uhat(num1) = exact_uhat_vec_on_face[i1];
      ++num1;
    }
  }
}

//
//

template <int dim, int spacedim>
void nargil::diffusion<dim, spacedim>::hdg_manager::run_interpolate_to_interior(
  nargil::cell<dim, spacedim> *in_cell, vectorFuncType func,
  nargil::distributed_vector<dim, spacedim> *out_vec)
{
  diffusion *own_cell = static_cast<diffusion *>(in_cell);
  const hdg_polybasis *own_basis =
    own_cell->template get_basis<hdg_polybasis>();
  hdg_manager *own_manager = own_cell->template get_manager<hdg_manager>();
  //
  unsigned n_scalar_unkns = own_basis->local_fe.base_element(0).dofs_per_cell;
  own_manager->exact_u.resize(n_scalar_unkns);
  own_manager->exact_q.resize(n_scalar_unkns * dim);
  //
  own_basis->local_fe_val_at_cell_supp->reinit(
    own_manager->my_dealii_local_dofs_cell);
  //
  for (unsigned i_unkn = 0; i_unkn < n_scalar_unkns; ++i_unkn)
  {
    dealii::Point<spacedim> q_point =
      own_basis->local_fe_val_at_cell_supp->quadrature_point(i_unkn);
    std::vector<double> value_at_i_node = func(q_point);
    own_manager->exact_u(i_unkn) = value_at_i_node[0];
    int idx1 = own_manager->local_interior_unkn_idx[i_unkn];
    out_vec->assemble(idx1, value_at_i_node[0]);
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
    {
      own_manager->exact_q(i_dim * n_scalar_unkns + i_unkn) =
        value_at_i_node[i_dim + 1];
      int idx2 =
        own_manager
          ->local_interior_unkn_idx[(i_dim + 1) * n_scalar_unkns + i_unkn];
      out_vec->assemble(idx2, value_at_i_node[i_dim + 1]);
    }
  }
}

//
//

template <int dim, int spacedim>
void nargil::diffusion<dim, spacedim>::hdg_manager::run_compute_local_unkns(
  nargil::cell<dim, spacedim> *in_cell)
{
  const diffusion *own_cell = static_cast<const diffusion *>(in_cell);
  //  hdg_polybasis *own_basis = static_cast<hdg_polybasis
  //  *>(own_cell->my_basis);
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(own_cell->my_manager.get());
  own_manager->compute_local_unkns();
}

//
//

// template <int dim, int spacedim>
// void nargil::diffusion<dim, spacedim>::hdg_manager::
//  run_assign_exact_local_unkns(
//    nargil::cell<dim, spacedim> *in_cell,
//    const dealii::parallel::distributed::Vector<double> &values)
//{
//  const diffusion *own_cell = static_cast<const diffusion *>(in_cell);
//  hdg_polybasis *own_basis = static_cast<hdg_polybasis *>(own_cell->my_basis);
//  hdg_manager *own_manager =
//    static_cast<hdg_manager *>(own_cell->my_manager.get());
//  unsigned n_scalar_unkns = own_basis->local_fe.base_element(0).dofs_per_cell;
//  dealii::Vector<double> dof_values(own_basis->local_fe.dofs_per_cell);
//  own_manager->my_dealii_local_dofs_cell->get_dof_values(values, dof_values);
//  own_manager->exact_u.resize(n_scalar_unkns);
//  own_manager->exact_q.resize(n_scalar_unkns * dim);
//  for (unsigned i1 = 0; i1 < n_scalar_unkns; ++i1)
//    own_manager->exact_u(i1) = dof_values[i1];
//  for (unsigned i1 = 0; i1 < dim * n_scalar_unkns; ++i1)
//    own_manager->exact_q(i1) = dof_values[i1 + n_scalar_unkns];
//}

//
//

template <int dim, int spacedim>
void nargil::diffusion<dim, spacedim>::hdg_manager::run_compute_errors(
  nargil::cell<dim, spacedim> *in_cell, std::vector<double> *sum_of_L2_errors)
{
  const diffusion *own_cell = static_cast<const diffusion *>(in_cell);
  hdg_polybasis *own_basis = static_cast<hdg_polybasis *>(own_cell->my_basis);
  hdg_manager *own_manager =
    static_cast<hdg_manager *>(own_cell->my_manager.get());
  unsigned n_scalar_unkns = own_basis->local_fe.base_element(0).dofs_per_cell;
  own_basis->local_fe_val_in_cell->reinit(
    own_manager->my_dealii_local_dofs_cell);
  for (unsigned i_quad = 0; i_quad < own_basis->cell_quad.size(); ++i_quad)
  {
    double JxW = own_basis->local_fe_val_in_cell->JxW(i_quad);
    for (unsigned i1 = 0; i1 < n_scalar_unkns; ++i1)
    {
      double shape_val_u =
        own_basis->local_fe_val_in_cell->shape_value(i1, i_quad);
      (*sum_of_L2_errors)[1] += 0;
    }
    for (unsigned i1 = n_scalar_unkns; i1 < (dim + 1) * n_scalar_unkns; ++i1)
    {
      unsigned i2 = i1 - n_scalar_unkns;
      double q_error = own_manager->exact_q(i2) - own_manager->q_vec(i2);

      //
      //
      //

      // std::cout << own_manager->q_vec << " " << own_manager->exact_q
      //          << std::endl;

      //
      //
      //
      double shape_val_q =
        own_basis->local_fe_val_in_cell->shape_value(i1, i_quad);
      //      std::cout << shape_val_q << std::endl;
      (*sum_of_L2_errors)[1] += shape_val_q * JxW * q_error * q_error;
    }
  }
}

template <int dim, int spacedim>
void nargil::diffusion<dim, spacedim>::hdg_manager::visualize_results(
  const dealii::DoFHandler<dim, spacedim> &dof_handler,
  const LA::MPI::Vector &visual_solu, const unsigned &time_level)
{
  const auto &tria = dof_handler.get_tria();
  unsigned n_active_cells = tria.n_active_cells();
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  unsigned refn_cycle = 0;
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  std::vector<std::string> solution_names(dim + 1);
  solution_names[0] = "head";
  for (unsigned i1 = 0; i1 < dim; ++i1)
    solution_names[i1 + 1] = "flow";
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      1, dealii::DataComponentInterpretation::component_is_scalar);
  for (unsigned i1 = 0; i1 < dim; ++i1)
    data_component_interpretation.push_back(
      dealii::DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector(visual_solu,
                           solution_names,
                           dealii::DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  dealii::Vector<float> subdomain(n_active_cells);
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = comm_rank;
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  const std::string filename =
    ("solution-" + dealii::Utilities::int_to_string(refn_cycle, 2) + "-" +
     dealii::Utilities::int_to_string(comm_rank, 4) + "-" +
     dealii::Utilities::int_to_string(time_level, 4));
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (comm_rank == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < comm_size; ++i)
      filenames.push_back(
        "solution-" + dealii::Utilities::int_to_string(refn_cycle, 2) + "-" +
        dealii::Utilities::int_to_string(i, 4) + "-" +
        dealii::Utilities::int_to_string(time_level, 4) + ".vtu");
    std::ofstream master_output((filename + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}
