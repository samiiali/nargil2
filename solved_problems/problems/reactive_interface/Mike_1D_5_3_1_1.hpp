#include <cstdio>
#include <fstream>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include <petsc.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/base/multithread_info.h>
#include <deal.II/lac/parallel_vector.h>

#include "include/elements/diffusion.hpp"
#include "include/elements/reactive_interface.hpp"
#include "include/mesh/mesh_handler.hpp"
#include "include/misc/utils.hpp"
#include "include/models/model.hpp"
#include "include/models/model_manager.hpp"
#include "include/ode_solvers/ode_solver.hpp"
#include "include/solvers/solvers.hpp"

///
///
/// Just a forward decleration.
///
///
template <int dim, int spacedim> struct RI_Problem1_dyna;

/**
 *
 * The data for diffusion problem:
 * \f[
 *   \Phi = \exp(\sin(x-y))
 * \f]
 * Hence, for \f$\mathbf E = -\lambda^2 \nabla \Phi \f$, we have:
 * \f[
 *   \mathbf E = 25(-\exp(\sin(x-y))\cos(x-y),\exp(sin(x-y))\cos(x-y)).
 * \f]
 *
 */
template <int dim, int spacedim = dim>
struct diff_data_2 : public nargil::diffusion<dim, spacedim>::data
{
  /**
   * @brief Constructor.
   */
  diff_data_2() : nargil::diffusion<dim, spacedim>::data() {}

  /**
   * @brief rhs_func.
   */
  virtual double rhs_func(const dealii::Point<spacedim> &p)
  {
    if (RI_Problem1_dyna<dim, spacedim>::in_semiconductor(p))
      return 0.;
    // if in electrolyte:
    return 0.;
  }

  /**
   * @brief gD_func.
   */
  virtual double gD_func(const dealii::Point<spacedim> &p)
  {
    if (p[0] > 1. - 1.e-8)
      return 0.0;
    else if (p[0] < -1. + 1.E-8)
      return 0.0;
    else
      assert(false);
  }

  /**
   * @brief gN_func.
   */
  virtual dealii::Tensor<1, dim> gN_func(const dealii::Point<spacedim> &p)
  {
    //
    return dealii::Tensor<1, dim>();
    //
    double x1 = p[0];
    double y1 = p[1];
    dealii::Tensor<1, dim> result(
      {-exp(sin(x1 - y1)) * cos(x1 - y1), exp(sin(x1 - y1)) * cos(x1 - y1)});
    return 25 * result;
  }

  /**
   * @brief exact_u
   */
  virtual double exact_u(const dealii::Point<spacedim> &p)
  {
    return 0.;
    //
    return exp(sin(p[0] - p[1]));
  }

  /**
   * @brief exact_q
   */
  virtual dealii::Tensor<1, dim> exact_q(const dealii::Point<spacedim> &p)
  {
    double x1 = p[0];
    double y1 = p[1];
    dealii::Tensor<1, dim> result(
      {-exp(sin(x1 - y1)) * cos(x1 - y1), exp(sin(x1 - y1)) * cos(x1 - y1)});
    //
    return dealii::Tensor<1, dim>();
    //
    return 25 * result;
  }

  /**
   *
   */
  virtual dealii::Tensor<2, dim> kappa_inv(const dealii::Point<spacedim> &p)
  {
    dealii::Tensor<2, dim> result;
    //
    // \lambda_S^2 = 1/l^* * sqrt((\Phi^* \epsilon_S \epsilon_0)/(q C^*))
    // \lambda_E^2 = 1/l^* * sqrt((\Phi^* \epsilon_E \epsilon_0)/(q C^*))
    //
    //
    // Values from page 72 of Mike diss:
    //
    // l^*     =  1.E-4        1/cm
    // \Phi^*  =  25.85e-3     V
    // t^*     =  1.E-12       s
    // C^*     =  1.E16        1/cm^3
    //
    // Values from page 162 of Mike diss:
    //
    // q          =  1.6E-19      C
    // \epsilon_S =  11.9
    // \epsilon_E =  1000
    //
    // lambda_E^2 =  0.378
    // lambda_S^2 =  0.0412
    //
    //
    // Obviously:
    //
    // kappa = lambda^2
    //
    //
    if (RI_Problem1_dyna<dim, spacedim>::in_semiconductor(p))
      result[0][0] = result[1][1] = 587.49;
    else if (RI_Problem1_dyna<dim, spacedim>::in_electrolyte(p))
      result[0][0] = result[1][1] = 6.9911;
    else
      assert(false);
    return result;
  }

  /**
   *
   */
  virtual double tau(const dealii::Point<spacedim> &)
  {
    //
    return 10.;
    //
  }
};

//
//
//
//
//

/**
 *
 * In this example we use the nargil::rereactive_interface element to solve a 2D
 * problem. Let us solve the reactive interface problem with the following
 * setup:
 *
 *
 *
 *       y=\pi/2     |-----------------|------------------|
 *                   |                 |                  |
 *           \Gamma_C|                 |\Sigma            |
 *                   |                 |                  |\Gamma_A
 *                   |     \Omega_S    |       \Omega_E   |
 *                   |                 |                  |
 *                   |                 |                  |
 *       y=-\pi/2    |-----------------|------------------|
 *                 x=-\pi             x=0               x=\pi
 *
 *
 *
 * Let us assume that:
 * \f[
 *   \begin{gather}
 *     \phi = \exp(\sin(x-y)), \quad
 *     \rho_n = \sin(x) + \cos(y), \quad
 *     \rho_p = \cos(x-y), \quad
 *     \rho_r = \sin(x+y), \quad
 *     \rho_o = \cos(x) - \sin(y).
 *   \end{gather}
 * \f]
 *
 */
template <int dim, int spacedim = dim>
struct react_int_problem1_data
  : public nargil::reactive_interface<dim, spacedim>::data
{
  /**
   * @brief pi
   */
  const double pi = M_PI;

  /**
   * @brief Constructor.
   */
  react_int_problem1_data() : nargil::reactive_interface<dim, spacedim>::data()
  {
  }

  /**
   * @brief rhs_func of \f$\rho_n\f$ equation.
   */
  virtual double rho_n_rhs_func(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    //
    return 1.0E-6;
    //
    return cos(x1) + sin(x1) +
           exp(sin(x1 - y1)) *
             (cos(x1 - y1) * (cos(x1) + cos(2 * x1 - y1) + cos(y1) +
                              (-1 + 2 * cos(x1 - y1)) * sin(x1)) -
              2 * (cos(x1) + sin(x1)) * sin(x1 - y1));
  }

  /**
   * @brief rhs_func of \f$\rho_n\f$ equation.
   */
  virtual double rho_p_rhs_func(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    //
    return 1.0E-6;
    //
    return 4 * cos(x1 - y1) *
           (1 + exp(sin(x1 - y1)) * (-pow(cos(x1 - y1), 2) + 2 * sin(x1 - y1)));
  }

  /**
   * @brief rhs_func of \f$\rho_r\f$ equation.
   */
  virtual double rho_r_rhs_func(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    //
    return 0;
    //
    return 6 * (1 + exp(sin(x1 - y1)) * (pow(cos(x1 - y1), 2) - sin(x1 - y1))) *
           sin(x1 + y1);
  }

  /**
   * @brief rhs_func of \f$\rho_o\f$ equation.
   */
  virtual double rho_o_rhs_func(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    //
    return 0;
    //
    return 4 * (cos(x1) - sin(y1) -
                ((cos(x1) + 4 * cos(x1 - 2 * y1) + cos(3 * x1 - 2 * y1) +
                  sin(2 * x1 - 3 * y1) - 4 * sin(2 * x1 - y1) - sin(y1)) *
                 (cosh(sin(x1 - y1)) + sinh(sin(x1 - y1)))) /
                  2.);
  }

  /**
   * @brief rhs_func of \f$\rho_n\f$ reactive interface.
   */
  virtual double rhs_of_RI_n(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];

    return 0.;

    return 6. * (-12. + cos(x1) + sin(x1)) * (cos(x1) - sin(y1));
  }

  /**
   * @brief rhs_func of \f$\rho_n\f$ reactive interface.
   */
  virtual double rhs_of_RI_p(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];

    return 0.;

    return 8. * (-22. + cos(x1 - y1)) * sin(x1 + y1);
  }

  /**
   * @brief rhs_func of \f$\rho_r\f$ reactive interface.
   */
  virtual double rhs_of_RI_r(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];

    return 0.;

    return 8. * (-22. + cos(x1 - y1)) * sin(x1 + y1) -
           6. * (-12. + cos(x1) + sin(x1)) * (cos(x1) - sin(y1));
  }

  /**
   * @brief rhs_func of \f$\rho_o\f$ reactive interface.
   */
  virtual double rhs_of_RI_o(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];

    return 0.;

    return 6. * (-12. + cos(x1) + sin(x1)) * (cos(x1) - sin(y1)) -
           8. * (-22. + cos(x1 - y1)) * sin(x1 + y1);
  }

  /**
   * @brief Dirichlet BC for \f$\rho_n\f$
   */
  virtual double gD_rho_n(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    // double y1 = p[1];
    //
    return rho_n_e();
    //
    return sin(x1) + cos(x1);
  }

  /**
   * @brief Dirichlet BC for \f$\rho_p\f$
   */
  virtual double gD_rho_p(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    //
    return rho_p_e();
    //
    return cos(x1 - y1);
  }

  /**
   * @brief Dirichlet BC for \f$\rho_r\f$
   */
  virtual double gD_rho_r(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    //
    return rho_r_inf();
    //
    return sin(x1 + y1);
  }

  /**
   * @brief Dirichlet BC for \f$\rho_o\f$
   */
  virtual double gD_rho_o(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    //
    return rho_o_inf();
    //
    return cos(x1) - sin(y1);
  }

  /**
   * @brief Dirichlet BC for \f$\rho_n\f$
   */
  virtual dealii::Tensor<1, dim>
  gN_rho_n(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    dealii::Tensor<1, dim> result(
      {-cos(x1) + sin(x1) +
         exp(sin(x1 - y1)) * cos(x1 - y1) * (cos(x1) + sin(x1)),
       -(exp(sin(x1 - y1)) * cos(x1 - y1) * (cos(x1) + sin(x1)))});

    return dealii::Tensor<1, dim>();

    return result;
  }

  /**
   * @brief Dirichlet BC for \f$\rho_p\f$
   */
  virtual dealii::Tensor<1, dim>
  gN_rho_p(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    dealii::Tensor<1, dim> result(
      {2 * (-(exp(sin(x1 - y1)) * pow(cos(x1 - y1), 2)) + sin(x1 - y1)),
       2 * exp(sin(x1 - y1)) * pow(cos(x1 - y1), 2) - 2 * sin(x1 - y1)});

    return dealii::Tensor<1, dim>();

    return result;
  }

  /**
   * @brief Dirichlet BC for \f$\rho_r\f$
   */
  virtual dealii::Tensor<1, dim>
  gN_rho_r(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    dealii::Tensor<1, dim> result(
      {-3 * cos(x1 + y1) +
         (3 * exp(sin(x1 - y1)) * (sin(2 * x1) + sin(2 * y1))) / 2.,
       -3 * (cos(x1 + y1) + exp(sin(x1 - y1)) * cos(x1 - y1) * sin(x1 + y1))});

    return dealii::Tensor<1, dim>();

    return result;
  }

  /**
   * @brief Dirichlet BC for \f$\rho_o\f$
   */
  virtual dealii::Tensor<1, dim>
  gN_rho_o(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    dealii::Tensor<1, dim> result(
      {4 * (sin(x1) - exp(sin(x1 - y1)) * cos(x1 - y1) * (cos(x1) - sin(y1))),
       4 * (cos(y1) + exp(sin(x1 - y1)) * cos(x1 - y1) * (cos(x1) - sin(y1)))});

    return dealii::Tensor<1, dim>();

    return result;
  }

  /**
   * @brief Initial values for \f$\rho_n\f$
   */
  virtual double rho_n_0(const dealii::Point<spacedim> &) final
  {
    return rho_n_e();
  }

  /**
   * @brief Initial values for \f$\rho_p\f$
   */
  virtual double rho_p_0(const dealii::Point<spacedim> &) final
  {
    return rho_p_e();
  }

  /**
   * @brief Initial values for \f$\rho_r\f$
   */
  virtual double rho_r_0(const dealii::Point<spacedim> &) final
  {
    return rho_r_inf();
  }

  /**
   * @brief Initial values for \f$\rho_o\f$
   */
  virtual double rho_o_0(const dealii::Point<spacedim> &) final
  {
    return rho_o_inf();
  }

  /**
   * @brief Initial values for \f$\rho_n\f$
   */
  virtual double rho_n_e() final { return 2.; }

  /**
   * @brief Initial value for \f$\rho_p\f$
   */
  virtual double rho_p_e() final { return 0.0; }

  /**
   * @brief Initial value for \f$\rho_r\f$
   */
  virtual double rho_r_inf() final { return 5.; }

  /**
   * @brief Initial value for \f$\rho_o\f$
   */
  virtual double rho_o_inf() final { return 4.; }

  /**
   *
   */
  virtual double lambda_inv2_S(
    const dealii::Point<spacedim> & = dealii::Point<spacedim>()) final
  {
    return 587.49;
  }

  /**
   *
   */
  virtual double
  lambda_inv2_E(const dealii::Point<spacedim> & = dealii::Point<spacedim>())
  {
    return 6.9911;
  }

  /**
   * @brief Value of \f$\mu_n\f$.
   */
  virtual double
  mu_n(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) final
  {
    return 3.4911E-3;
  }

  /**
   * @brief Value of \f$\mu_p\f$.
   */
  virtual double
  mu_p(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) final
  {
    return 1.2413E-3;
  }

  /**
   * @brief Value of \f$\mu_r\f$.
   */
  virtual double
  mu_r(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) final
  {
    return 5.172E-4;
  }

  /**
   * @brief Value of \f$\mu_o\f$.
   */
  virtual double
  mu_o(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) final
  {
    return 5.172E-4;
  }

  /**
   * @brief Value of intrinsic electron density.
   */
  virtual double intrinsic_rho() final { return 2.564e-7; }

  /**
   * @brief Value of rescaled electron lifetime.
   */
  virtual double tau_n() final { return 5.e7; }

  /**
   * @brief Value of rescaled hole lifetime.
   */
  virtual double tau_p() final { return 5.e7; }

  /**
   * @brief Value of \f$k_{ht}\f$.
   */
  virtual double k_et() final { return 1.e-11; }

  /**
   * @brief Value of \f$k_{ht}\f$.
   */
  virtual double k_ht() final { return 1.e-6; }

  /**
   * @brief Value of \f$\mu_n\f$.
   */
  virtual double
  alpha_n(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) final
  {
    return -1.0;
  }

  /**
   * @brief Value of \f$\mu_p\f$.
   */
  virtual double
  alpha_p(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) final
  {
    return 1.0;
  }

  /**
   * @brief Value of \f$\mu_r\f$.
   */
  virtual double
  alpha_r(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) final
  {
    return -1.0;
  }

  /**
   * @brief Value of \f$\mu_o\f$.
   */
  virtual double
  alpha_o(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) final
  {
    return 1.0;
  }

  /**
   * @brief exact_u
   */
  virtual double exact_rho_n(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    //
    return 0.;
    //
    return sin(x1) + cos(x1);
  }

  /**
   * @brief exact_u
   */
  virtual double exact_rho_p(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    //
    return 0.;
    //
    return cos(x1 - y1);
  }

  /**
   * @brief exact_u
   */
  virtual double exact_rho_r(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    //
    return 0.;
    //
    return sin(x1 + y1);
  }

  /**
   * @brief exact_u
   */
  virtual double exact_rho_o(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    //
    return 0.;
    //
    return cos(x1) - sin(y1);
  }

  /**
   *
   */
  virtual dealii::Tensor<1, dim>
  exact_q_n(const dealii::Point<spacedim> &p) final
  {
    dealii::Tensor<1, dim> result({-cos(p[0]) + sin(p[0]), 0.0});
    //
    return dealii::Tensor<1, dim>();
    //
    return result;
  }

  /**
   *
   */
  virtual dealii::Tensor<1, dim>
  exact_q_p(const dealii::Point<spacedim> &p) final
  {
    dealii::Tensor<1, dim> result(
      {2 * sin(p[0] - p[1]), -2 * sin(p[0] - p[1])});
    //
    return dealii::Tensor<1, dim>();
    //
    return result;
  }

  /**
   *
   */
  virtual dealii::Tensor<1, dim>
  exact_q_r(const dealii::Point<spacedim> &p) final
  {
    dealii::Tensor<1, dim> result(
      {-3 * cos(p[0] + p[1]), -3 * cos(p[0] + p[1])});
    //
    return dealii::Tensor<1, dim>();
    //
    return result;
  }

  /**
   *
   */
  virtual dealii::Tensor<1, dim>
  exact_q_o(const dealii::Point<spacedim> &p) final
  {
    dealii::Tensor<1, dim> result({4 * sin(p[0]), 4 * cos(p[1])});
    //
    return dealii::Tensor<1, dim>();
    //
    return result;
  }

  /**
   * @brief Electric field.
   */
  virtual dealii::Tensor<1, dim>
  electric_field(const dealii::Point<spacedim> &p) final
  {
    double x1 = p[0];
    double y1 = p[1];
    dealii::Tensor<1, dim> result({-25 * exp(sin(x1 - y1)) * cos(x1 - y1),
                                   25 * exp(sin(x1 - y1)) * cos(x1 - y1)});
    return result;
  }

  /**
   * @brief the stabilization parameter.
   */
  virtual double tau(const dealii::Point<spacedim> &) final { return 1000.0; }
};

/**
 * Just a sample problem
 */
template <int dim, int spacedim = dim> struct RI_Problem1_dyna
{
  typedef nargil::reactive_interface<dim> R_I_Eq;
  typedef nargil::model<R_I_Eq, dim> R_I_Model;
  typedef typename R_I_Eq::hdg_polybasis R_I_BasisType;
  typedef typename nargil::reactive_interface<dim>::template hdg_manager<
    R_I_BasisType>
    R_I_ManagerType;

  typedef nargil::diffusion<dim> DiffEq;
  typedef nargil::model<DiffEq, dim> DiffModel;
  typedef typename DiffEq::hdg_polybasis DiffBasis;
  typedef typename nargil::diffusion<dim>::template hdg_manager<DiffBasis>
    DiffManagerType;

  /**
   *
   */
  static bool in_semiconductor(const dealii::Point<spacedim> &p)
  {
    if (p[0] < 1.E-8)
      return true;
    return false;
  }

  /**
   *
   */
  static bool in_electrolyte(const dealii::Point<spacedim> &p)
  {
    if (p[0] > -1.E-8)
      return true;
    return false;
  }

  /**
   * @brief adaptive_mesh_gen
   */
  static void mesh_gen_0(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    std::vector<unsigned> refine_repeats = {80, 40};
    dealii::Point<dim> corner_1(-M_PI, -M_PI / 2.);
    dealii::Point<dim> corner_2(M_PI, M_PI / 2.);
    dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, refine_repeats,
                                                      corner_1, corner_2, true);
    // Adding periodicity.
    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::parallel::distributed::Triangulation<
        dim>::cell_iterator> >
      periodic_faces;
    dealii::GridTools::collect_periodic_faces(
      the_mesh, 0, 1, 0, periodic_faces,
      dealii::Tensor<1, dim>({2. * M_PI, 0.}));
    dealii::GridTools::collect_periodic_faces(
      the_mesh, 2, 3, 0, periodic_faces, dealii::Tensor<1, dim>({0., M_PI}));
    the_mesh.add_periodicity(periodic_faces);
  }

  /**
   *
   */
  static void mesh_gen_1(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    std::vector<dealii::Point<dim> > vertices_S, vertices_E;
    //
    vertices_S.push_back(dealii::Point<dim>(0., 0.));
    vertices_S.push_back(dealii::Point<dim>(0.7, 0.));
    vertices_S.push_back(dealii::Point<dim>(0., 1.));
    vertices_S.push_back(dealii::Point<dim>(0.3, 1.));
    vertices_E.push_back(dealii::Point<dim>(0.7, 0.));
    vertices_E.push_back(dealii::Point<dim>(1.0, 0.));
    vertices_E.push_back(dealii::Point<dim>(0.3, 1.));
    vertices_E.push_back(dealii::Point<dim>(1., 1.));
    //
    dealii::parallel::distributed::Triangulation<dim, spacedim> mesh1(
      MPI_COMM_WORLD);
    dealii::parallel::distributed::Triangulation<dim, spacedim> mesh2(
      MPI_COMM_WORLD);
    dealii::GridGenerator::general_cell(mesh1, vertices_S, false);
    dealii::GridGenerator::general_cell(mesh2, vertices_E, false);
    dealii::GridGenerator::merge_triangulations(mesh1, mesh2, the_mesh);
    the_mesh.refine_global(5);
  }

  //
  //

  static void mesh_gen(
    dealii::parallel::distributed::Triangulation<dim, spacedim> &the_mesh)
  {
    std::vector<unsigned> refine_repeats = {200, 1};
    dealii::Point<dim> corner_1(-1., 0.);
    dealii::Point<dim> corner_2(1., 1.E-2);
    dealii::GridGenerator::subdivided_hyper_rectangle(the_mesh, refine_repeats,
                                                      corner_1, corner_2, true);
  }

  //
  //

  /**
   *
   * @brief dofs_on_nodes
   *
   */
  static void assign_diff_BCs(DiffManagerType *in_manager)
  {
    unsigned n_dof_per_face = DiffBasis::get_n_dofs_per_face();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      auto &&face = in_manager->my_cell->my_dealii_cell->face(i_face);
      if (face->at_boundary())
      {
        if (face->center()[0] > 1. - 1.E-8 || face->center()[0] < -1. + 1.E-8)
        {
          in_manager->BCs[i_face] =
            nargil::diffusion<dim>::boundary_condition::essential;
          in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 0);
        }
        else
        {
          in_manager->BCs[i_face] =
            nargil::diffusion<dim>::boundary_condition::natural;
          in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 1);
        }
      }
      else
      {
        in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 1);
      }
    }
  }

  //
  //

  /**
   *
   * For this problem we set the elements on the postive x to have essential
   * boundary condition on \f$\rho_n, \rho_p\f$ and the elements on the
   * negative x to have essential BC on \f$\rho_r, \rho_o\f$. For elements
   * in the electrolyte, we use not_set BC with restrained \f$\rho_n, \rho_p\f$.
   * For elements in semi-conductor, we use not_set BC with restrained
   * \f$\rho_r, \rho_o\f$.
   *
   */
  static void assign_R_I_BCs(R_I_ManagerType *in_manager)
  {
    unsigned n_dof_per_face = R_I_BasisType::get_n_dofs_per_face();
    in_manager->local_equation_is_active.resize(4, 0);
    dealii::Point<spacedim> cell_center =
      in_manager->my_cell->my_dealii_cell->center();
    if (in_semiconductor(cell_center)) // semiconductor
    {
      in_manager->local_equation_is_active[0] =
        in_manager->local_equation_is_active[1] = 1;
    }
    if (in_electrolyte(cell_center)) // electrolyte
    {
      in_manager->local_equation_is_active[2] =
        in_manager->local_equation_is_active[3] = 1;
    }

    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      auto &&face = in_manager->my_cell->my_dealii_cell->face(i_face);
      in_manager->BCs[i_face] = R_I_Eq::boundary_condition::not_set;
      dealii::Point<dim> face_center = face->center();
      if (in_semiconductor(face_center)) // We are in semi-conductor
      {
        in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 0);
        in_manager->dof_status_on_faces[i_face][0] = 1;
        in_manager->dof_status_on_faces[i_face][1] = 1;
        //
        // Applying BC on rho_n and rho_p
        //
        if (std::abs(face_center[0] + 1.) < 1.E-8)
        {
          in_manager->BCs[i_face] =
            static_cast<typename R_I_Eq::boundary_condition>(
              in_manager->BCs[i_face] |
              R_I_Eq::boundary_condition::essential_rho_n |
              R_I_Eq::boundary_condition::essential_rho_p);
          in_manager->dof_status_on_faces[i_face][0] = 0;
          in_manager->dof_status_on_faces[i_face][1] = 0;
        }
        if (std::abs(face_center[0]) < 1.E-8)
        {
          in_manager->BCs[i_face] =
            static_cast<typename R_I_Eq::boundary_condition>(
              in_manager->BCs[i_face] |
              R_I_Eq::boundary_condition::semiconductor_R_I);
          in_manager->dof_status_on_faces[i_face][0] = 1;
          in_manager->dof_status_on_faces[i_face][1] = 1;
        }
        if (std::abs(face_center[1]) < 1.E-8 ||
            std::abs(face_center[1] - 1.E-2) < 1E-8)
        {
          in_manager->BCs[i_face] =
            static_cast<typename R_I_Eq::boundary_condition>(
              in_manager->BCs[i_face] |
              R_I_Eq::boundary_condition::natural_rho_n |
              R_I_Eq::boundary_condition::natural_rho_p);
          in_manager->dof_status_on_faces[i_face][0] = 1;
          in_manager->dof_status_on_faces[i_face][1] = 1;
        }
      }
      if (in_electrolyte(face_center)) // We are in electrolyte
      {
        in_manager->dof_status_on_faces[i_face].resize(n_dof_per_face, 0);
        in_manager->dof_status_on_faces[i_face][2] = 1;
        in_manager->dof_status_on_faces[i_face][3] = 1;
        //
        // Applying BC on rho_r and rho_o
        //
        if (std::abs(face_center[0] - 1.) < 1.E-8)
        {
          in_manager->BCs[i_face] =
            static_cast<typename R_I_Eq::boundary_condition>(
              in_manager->BCs[i_face] |
              R_I_Eq::boundary_condition::essential_rho_r |
              R_I_Eq::boundary_condition::essential_rho_o);
          in_manager->dof_status_on_faces[i_face][2] = 0;
          in_manager->dof_status_on_faces[i_face][3] = 0;
        }
        if (std::abs(face_center[0]) < 1.E-8)
        {
          in_manager->BCs[i_face] =
            static_cast<typename R_I_Eq::boundary_condition>(
              in_manager->BCs[i_face] |
              R_I_Eq::boundary_condition::electrolyte_R_I);
          in_manager->dof_status_on_faces[i_face][2] = 1;
          in_manager->dof_status_on_faces[i_face][3] = 1;
        }
        if (std::abs(face_center[1]) < 1E-8 ||
            std::abs(face_center[1] - 1.E-2) < 1.E-8)
        {
          in_manager->BCs[i_face] =
            static_cast<typename R_I_Eq::boundary_condition>(
              in_manager->BCs[i_face] |
              R_I_Eq::boundary_condition::natural_rho_r |
              R_I_Eq::boundary_condition::natural_rho_o);
          in_manager->dof_status_on_faces[i_face][2] = 1;
          in_manager->dof_status_on_faces[i_face][3] = 1;
        }
      }
    }
  }

  //
  //

  static void run(int argc, char **argv)
  {
    static_assert(dim == 2, "dim should be equal to 3.");

    PetscInitialize(&argc, &argv, NULL, NULL);
    dealii::MultithreadInfo::set_thread_limit(1);
    //
    // PETSc scope.
    //
    {
      double dt = 0.01;
      const MPI_Comm &comm = PETSC_COMM_WORLD;
      int comm_rank, comm_size;
      MPI_Comm_rank(comm, &comm_rank);
      MPI_Comm_size(comm, &comm_size);
      //
      nargil::mesh<dim> mesh1(comm, 1, true);
      mesh1.generate_mesh(mesh_gen);
      //
      diff_data_2<dim> data0;
      react_int_problem1_data<dim> data1;
      //
      DiffBasis basis0(1, 2);
      nargil::implicit_hybridized_numbering<dim> dof_counter0;
      nargil::hybridized_model_manager<dim> model_manager0;
      //
      R_I_BasisType basis1(1, 2);
      nargil::implicit_hybridized_numbering<dim> dof_counter1;
      nargil::hybridized_model_manager<dim> model_manager1;
      //
      nargil::ode_solvers::trapezoidal_solver<Eigen::MatrixXd> tpz_integrator1(
        dt);
      //
      // Mesh refinement cycle.
      //
      unsigned i_cycle = 0;
      //
      mesh1.init_cell_ID_to_num();
      //
      // Initializing models.
      //
      DiffModel model0(mesh1);
      model0.init_model_elements(&basis0);
      model_manager0.form_dof_handlers(&model0, &basis0);
      //
      R_I_Model model1(mesh1);
      model1.init_model_elements(&basis1);
      model_manager1.form_dof_handlers(&model1, &basis1);
      //
      // Applying BCs.
      //
      model_manager0.apply_on_owned_cells(&model0, DiffManagerType::assign_BCs,
                                          assign_diff_BCs);
      model_manager0.apply_on_ghost_cells(&model0, DiffManagerType::assign_BCs,
                                          assign_diff_BCs);
      //
      model_manager1.apply_on_owned_cells(&model1, R_I_ManagerType::assign_BCs,
                                          assign_R_I_BCs);
      model_manager1.apply_on_ghost_cells(&model1, R_I_ManagerType::assign_BCs,
                                          assign_R_I_BCs);
      //
      // Counting global unknowns
      //
      dof_counter0.template count_globals<DiffBasis>(&model0);
      dof_counter1.template count_globals<R_I_BasisType>(&model1);
      //
      // Connecting R_I model with Diffusion model.
      //
      model_manager0.connect_to_other_model(&model0, &model1);
      model_manager1.connect_to_other_model(&model1, &model0);
      //
      // Setting source terms and the contribution of BCs.
      //
      model_manager0.apply_on_owned_cells(&model0, DiffEq::assign_data, &data0);
      model_manager1.apply_on_owned_cells(&model1, R_I_Eq::assign_data, &data1);
      //
      // Getting the solvers ready.
      //
      int solver_keys0 = nargil::solvers::solver_props::spd_matrix;
      int update_keys0 = nargil::solvers::solver_update_opts::update_mat |
                         nargil::solvers::solver_update_opts::update_rhs;
      nargil::solvers::petsc_direct_solver<dim> solver0(solver_keys0,
                                                        dof_counter0, comm);
      //
      int solver_keys1 = nargil::solvers::solver_props::default_option;
      int update_keys1 = nargil::solvers::solver_update_opts::update_mat |
                         nargil::solvers::solver_update_opts::update_rhs;
      nargil::solvers::petsc_direct_solver<dim> solver1(solver_keys1,
                                                        dof_counter1, comm);
      //
      // This might be a little overkill.
      //
      model_manager1.apply_on_owned_cells(
        &model1, R_I_ManagerType::set_time_integrator, &tpz_integrator1,
        R_I_ManagerType::time_integrator_type::TRPZ);
      //
      // Setting the initial value of the R-I problem.
      //
      model_manager1.apply_on_owned_cells(&model1,
                                          R_I_ManagerType::set_source_and_BCs);
      model_manager1.apply_on_owned_cells(&model1,
                                          R_I_ManagerType::set_init_vals);
      //
      // The time stepping loop.
      //
      for (unsigned i_time = 0; i_time < 200; ++i_time)
      {
        //
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::set_source_and_BCs);
        model_manager0.apply_on_owned_cells(&model0,
                                            DiffManagerType::compute_matrices);
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::template apply_R_I_source<R_I_ManagerType>);
        //
        solver0.reinit_components(solver_keys0, update_keys0);
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::assemble_globals, &solver0);
        //
        // Distributing the solution of diffusion problem among cells.
        //
        Vec sol_vec0;
        solver0.finish_assemble(update_keys0);
        solver0.form_factors();
        solver0.solve_system(&sol_vec0);
        //
        std::vector<double> local_sol_vec0(
          solver0.get_local_part_of_global_vec(&sol_vec0));
        //
        // Computing the interior unknowns.
        //
        model_manager0.apply_on_owned_cells(&model0,
                                            DiffManagerType::compute_matrices);
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::template apply_R_I_source<R_I_ManagerType>);
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::compute_local_unkns, local_sol_vec0.data());
        //
        nargil::distributed_vector<dim> dist_sol_vec0(
          model_manager0.local_dof_handler, PETSC_COMM_WORLD);
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::fill_viz_vector, &dist_sol_vec0);
        //
        LA::MPI::Vector global_sol_vec0;
        dist_sol_vec0.copy_to_global_vec(global_sol_vec0);
        //
        // We prepare the visulization data and visualize the results.
        //
        typename DiffEq::viz_data viz_data0(
          comm, &model_manager0.local_dof_handler, &global_sol_vec0, "diff_sol",
          "Head", "Flow");
        viz_data0.cycle = i_cycle;
        viz_data0.time_step = i_time;
        DiffManagerType::visualize_results(viz_data0);
        //
        // We interpolated exact u and q to u_exact and q_exact
        //
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::interpolate_to_interior);
        //
        std::vector<double> sum_of_diff_L2_errors(2, 0);
        model_manager0.apply_on_owned_cells(
          &model0, DiffManagerType::compute_errors, &sum_of_diff_L2_errors);
        double u_error_global, q_error_global;
        MPI_Reduce(&sum_of_diff_L2_errors[0], &u_error_global, 1, MPI_DOUBLE,
                   MPI_SUM, 0, comm);
        MPI_Reduce(&sum_of_diff_L2_errors[1], &q_error_global, 1, MPI_DOUBLE,
                   MPI_SUM, 0, comm);
        //
        if (comm_rank == 0)
        {
          constexpr unsigned str_size = 200;
          char accuracy_output[str_size];
          snprintf(accuracy_output, str_size,
                   "The errors are: \n"
                   "diffusion u error: \033[3;33m %10.4E \033[0m\n"
                   "diffusion q error: \033[3;33m  %10.4E \033[0m\n",
                   sqrt(u_error_global), sqrt(q_error_global));
          // ***
          // std::cout << accuracy_output << std::endl;
        }

        //
        //
        // Now solving the R-I problem.
        //
        //

        model_manager1.apply_on_owned_cells(
          &model1, R_I_ManagerType::set_source_and_BCs);
        //
        // Now, we get the solution from the diffusion element.
        //
        model_manager1.apply_on_owned_cells(
          &model1,
          R_I_ManagerType::template get_E_from_relevant_cell<DiffManagerType>);
        //
        // This is Newton-Raphson iteration loop
        //
        double NR_delta;
        double NR_cycle = 0;
        do
        {
          ++NR_cycle;
          solver1.reinit_components(solver_keys1, update_keys1);
          //
          model_manager1.apply_on_owned_cells(
            &model1, R_I_ManagerType::compute_linear_matrices);
          model_manager1.apply_on_owned_cells(
            &model1, R_I_ManagerType::add_nonlinear_terms);
          model_manager1.apply_on_owned_cells(
            &model1, R_I_ManagerType::add_recombination_source);
          model_manager1.apply_on_owned_cells(
            &model1, R_I_ManagerType::assemble_globals, &solver1);
          //
          Vec sol_vec1;
          solver1.finish_assemble(update_keys1);
          solver1.form_factors();
          solver1.solve_system(&sol_vec1);
          //
          std::vector<double> local_sol_vec1(
            solver1.get_local_part_of_global_vec(&sol_vec1));
          model_manager1.apply_on_owned_cells(
            &model1, R_I_ManagerType::extract_NR_increment,
            local_sol_vec1.data());
          model_manager1.apply_on_owned_cells(
            &model1, R_I_ManagerType::compute_NR_increments);
          //
          // Now, we check if the NR scheme is converged. Here, we should do an
          // MPI_Allreduce, because we need all ranks to have the sum.
          //
          std::vector<double> sum_of_NR_delta(4, 0);
          model_manager1.apply_on_owned_cells(
            &model1, R_I_ManagerType::get_L2_norm_of_NR_deltas,
            &sum_of_NR_delta);
          //
          std::vector<double> global_NR_delta(4, 0.);
          for (unsigned i1 = 0; i1 < 4; ++i1)
            MPI_Allreduce(&sum_of_NR_delta[i1], &global_NR_delta[i1], 1,
                          MPI_DOUBLE, MPI_SUM, comm);
          //
          NR_delta = std::accumulate(global_NR_delta.begin(),
                                     global_NR_delta.end(), 0.0);
          NR_delta = sqrt(NR_delta);
          if (comm_rank == 0)
          {
            char accuracy_output[400];
            snprintf(accuracy_output, 400,
                     "The L2 norm of Newton-Raphson increment is: "
                     "\033[3;33m %10.4E \033[0m",
                     NR_delta);
            std::cout << "Newton-Raphson iteration " << NR_cycle << ": "
                      << accuracy_output << std::endl;
          }
        } while (NR_delta > 1.e-8 && NR_cycle < 20);
        //
        //
        //
        nargil::distributed_vector<dim> dist_sol_vec(
          model_manager1.viz_dof_handler, PETSC_COMM_WORLD);
        //
        model_manager1.apply_on_owned_cells(
          &model1, R_I_ManagerType::fill_viz_vector, &dist_sol_vec);
        //
        LA::MPI::Vector global_sol_vec;
        dist_sol_vec.copy_to_global_vec(global_sol_vec);
        //
        // Now we visualize the results
        //
        std::vector<std::string> var_names({"rho_n", "rho_n_flow", "rho_p",
                                            "rho_p_flow", "rho_r", "rho_r_flow",
                                            "rho_o", "rho_o_flow"});
        typename R_I_Eq::viz_data viz_data1(
          comm, &model_manager1.viz_dof_handler, &global_sol_vec, "R_I_sol",
          var_names);
        viz_data1.time_step = i_time;
        viz_data1.cycle = i_cycle;
        R_I_ManagerType::visualize_results(viz_data1);
        //
        // Now we want to compute the errors.
        //
        model_manager1.apply_on_owned_cells(
          &model1, R_I_ManagerType::interpolate_to_interior);
        std::vector<double> sum_of_L2_errors(8, 0);
        model_manager1.apply_on_owned_cells(
          &model1, R_I_ManagerType::compute_errors, &sum_of_L2_errors);

        std::vector<double> global_errors(8, 0);
        for (unsigned i1 = 0; i1 < 8; ++i1)
          MPI_Reduce(&sum_of_L2_errors[i1], &global_errors[i1], 1, MPI_DOUBLE,
                     MPI_SUM, 0, comm);

        if (comm_rank == 0)
        {
          char accuracy_output[400];
          snprintf(accuracy_output, 400,
                   // ***
                   //       "The errors are: \n"
                   "The L2 norms of the solutions are: \n"
                   "rho_n error: \033[3;33m %10.4E \033[0m\n"
                   "q_n error: \033[3;33m  %10.4E \033[0m\n"
                   "rho_p error: \033[3;33m %10.4E \033[0m\n"
                   "q_p error: \033[3;33m  %10.4E \033[0m\n"
                   "rho_r error: \033[3;33m %10.4E \033[0m\n"
                   "q_r error: \033[3;33m  %10.4E \033[0m\n"
                   "rho_o error: \033[3;33m %10.4E \033[0m\n"
                   "q_o error: \033[3;33m  %10.4E \033[0m\n",
                   sqrt(global_errors[0]), sqrt(global_errors[1]),
                   sqrt(global_errors[2]), sqrt(global_errors[3]),
                   sqrt(global_errors[4]), sqrt(global_errors[5]),
                   sqrt(global_errors[6]), sqrt(global_errors[7]));
          std::cout << accuracy_output << std::endl;
        }
      }
    }
    //
    // PETSc scope.
    //
    PetscFinalize();
  }
};
