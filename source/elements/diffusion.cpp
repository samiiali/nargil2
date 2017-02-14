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
  dealii::UpdateFlags fe_face_val_flags(dealii::update_values |
                                        dealii::update_JxW_values |
                                        dealii::update_quadrature_points);

  std::unique_ptr<dealii::FEValues<dim> > local_fe_vel_temp(
    new dealii::FEValues<dim>(local_fe, cell_quad, fe_val_flags));
  std::unique_ptr<dealii::FEFaceValues<dim> > trace_fe_face_val_temp(
    new dealii::FEFaceValues<dim>(trace_fe, face_quad, fe_face_val_flags));

  local_fe_val_in_cell = std::move(local_fe_vel_temp);
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
template <typename Func>
std::vector<double>
nargil::diffusion<dim, spacedim>::hdg_manager::interpolate_to_trace_unkns(
  const Func func, const unsigned i_face)
{
  const diffusion *own_cell = static_cast<const diffusion *>(this->my_cell);
  hdg_polybasis *own_basis = static_cast<hdg_polybasis *>(own_cell->my_basis);
  std::vector<double> dof_values(own_basis->trace_fe.dofs_per_face);
  own_basis->trace_fe_face_val->reinit(this->my_dealii_trace_dofs_cell, i_face);
  const std::vector<dealii::Point<spacedim> > &face_quad_locs =
    own_basis->trace_fe_face_val->get_quadrature_points();
  std::cout << face_quad_locs[0] << std::endl;
  std::transform(face_quad_locs.begin(), face_quad_locs.end(),
                 dof_values.begin(), func);
  return dof_values;
}

//
//

template <int dim, int spacedim>
void nargil::diffusion<dim, spacedim>::hdg_manager::set_trace_unkns(
  const std::vector<double> &values)
{
  const diffusion *own_cell = static_cast<const diffusion *>(this->my_cell);
  hdg_polybasis *own_basis = static_cast<hdg_polybasis *>(own_cell->my_basis);
  assert(values.size() == own_basis->trace_fe.dofs_per_cell);
  exact_uhat.resize(values.size());
  for (unsigned i1 = 0; i1 < values.size(); ++i1)
    exact_uhat(i1) = values[i1];
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
  const diffusion *own_cell = static_cast<const diffusion *>(this->my_cell);
  hdg_polybasis *own_basis = static_cast<hdg_polybasis *>(own_cell->my_basis);
  own_basis->local_fe_val_in_cell->reinit(this->my_dealii_local_dofs_cell);

  compute_matrices();

  //  own_basis->local_fe_val_in_cell->
}

//
//

template <int dim, int spacedim>
std::vector<double>
nargil::diffusion<dim, spacedim>::hdg_manager::compute_local_errors(
  const dealii::Function<dim> &exact_sol_func)
{

  const diffusion *own_cell = static_cast<const diffusion *>(this->my_cell);
  hdg_polybasis *own_basis = static_cast<hdg_polybasis *>(own_cell->my_basis);
  unsigned n_blocks = own_basis->local_fe_val_in_cell.n_blocks();
  std::vector<double> error(n_blocks, 0.);
  own_basis->local_fe_val_in_cell->reinit(this->my_dealii_local_dofs_cell);

  compute_matrices();

  //  own_basis->local_fe_val_in_cell->
  return error;
}

//
//

template <int dim, int spacedim>
std::function<double(dealii::Point<spacedim>)>
  nargil::diffusion<dim, spacedim>::hdg_manager::my_exact_uhat_func;

//
//

template <int dim, int spacedim>
template <typename Func>
void nargil::diffusion<dim, spacedim>::hdg_manager::set_exact_uhat_func(Func f)
{
  my_exact_uhat_func = f;
}

//
//

template <int dim, int spacedim>
void nargil::diffusion<dim, spacedim>::hdg_manager::compute_matrices()
{
  const diffusion *own_cell = static_cast<const diffusion *>(this->my_cell);
  hdg_polybasis *own_basis = static_cast<hdg_polybasis *>(own_cell->my_basis);
  unsigned n_scalar_unkns = pow(own_basis->_poly_order + 1, dim);
  //  unsigned n_trace_unkns = pow(my_own_basis->_poly_order + 1, dim);

  dealii::FEValuesExtractors::Scalar scalar(0);
  dealii::FEValuesExtractors::Vector fluxes(1);

  unsigned n_trace_unkns_per_face = own_basis->trace_fe.dofs_per_face;

  //  unsigned n_u_unkns = my_own_basis->u_basis.dofs_per_cell;

  //  my_own_basis->u_in_cell.reinit(my_own_cell->my_dealii_cell);
  (*own_basis->local_fe_val_in_cell).reinit(this->my_dealii_local_dofs_cell);

  A = Eigen::MatrixXd::Zero(dim * n_scalar_unkns, dim * n_scalar_unkns);
  B = Eigen::MatrixXd::Zero(dim * n_scalar_unkns, n_scalar_unkns);

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

  //  for (unsigned i_unkn = 0; i_unkn < n_local_unkns_per_cell; ++i_unkn)
  //  {
  //    for (unsigned j_unkn = 0; j_unkn < n_local_unkns_per_cell; ++j_unkn)
  //    {
  //      for (unsigned i_quad = 0; i_quad < own_basis->cell_quad.size();
  //      ++i_quad)
  //      {
  //        if (i_unkn < n_scalar_unkns && j_unkn < n_scalar_unkns)
  //        {
  //        }
  //        if (i_unkn >= n_scalar_unkns && j_unkn >= n_scalar_unkns)
  //        {
  //        }
  //        if (i_unkn >= n_scalar_unkns && j_unkn < n_scalar_unkns)
  //        {
  //        }
  //        auto aaa =
  //          (*own_basis->local_fe_val_in_cell)[fluxes].value(i_unkn,
  //          i_quad);
  //        std::cout << aaa << std::endl;
  //      }
  //    }
  //  }

  unsigned i_face_unkn = 0;
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    own_basis->trace_fe_face_val->reinit(this->my_dealii_trace_dofs_cell,
                                         i_face);
    for (unsigned i_each_face_unkn = 0;
         i_each_face_unkn < n_trace_unkns_per_face;
         ++i_each_face_unkn)
    {
      for (unsigned i_face_quad = 0; i_face_quad < own_basis->face_quad.size();
           ++i_face_quad)
      {
        /*
        auto aaa = my_own_basis->trace_fe_face_val->shape_value(i_face_unkn,
                                                                i_face_quad);
        std::cout << aaa << std::endl;
        */
      }
      ++i_face_unkn;
    }
  }
}

//
//

template <int dim, int spacedim>
void nargil::diffusion<dim, spacedim>::hdg_manager::
  run_interpolate_and_set_uhat(nargil::cell<dim, spacedim> *in_cell)
{
  diffusion *own_cell = static_cast<diffusion *>(in_cell);
  const hdg_polybasis *own_basis =
    own_cell->template get_basis<hdg_polybasis>();
  auto i_manager = own_cell->template get_manager<hdg_manager>();
  std::vector<double> exact_uhat_vec(own_basis->trace_fe.n_dofs_per_cell());
  unsigned num1 = 0;
  for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
  {
    std::vector<double> exact_uhat_vec_on_face =
      i_manager->interpolate_to_trace_unkns(my_exact_uhat_func, i_face);
    for (unsigned i1 = 0; i1 < own_basis->trace_fe.n_dofs_per_face(); ++i1)
    {
      exact_uhat_vec[num1] = exact_uhat_vec_on_face[i1];
      ++num1;
    }
  }
  i_manager->set_trace_unkns(exact_uhat_vec);
}
