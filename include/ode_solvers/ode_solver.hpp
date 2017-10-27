#ifndef ODE_SOLVERS
#define ODE_SOLVERS

namespace nargil
{
namespace ode_solvers
{
/**
 *
 *
 * @brief This is the base struct for all other 1st order ODE solvers.
 *
 * The base class for all other 1st order ODE solvers. We want this
 * class to be polymorphic. So, the desctructor is defined virtual.
 *
 *
 */
struct first_order_ode_solver
{
  first_order_ode_solver() {}
  virtual ~first_order_ode_solver() {}
};

/**
 *
 * This structure solves the ODE \f$y'(t) = f(t,y)\f$. Having
 * \f$y_0\f$ as the solution in the last time step and \f$y_1\f$ as
 * the solution in the current time step, and \f$h\f$ as time step size,
 * we satisfy: \f$y_1 - y_0 = h f(t_1,y_1)\f$.
 *
 * For example, refer to reactive_interface::hdg_manager, where we have
 * the equaiton: \f$\partial_t \rho_n + \nabla \cdot (\mu_n \alpha_n
 * \lambda^{-2} \mathbf E \rho_n) + \nabla \cdot \mathbf q_n = f_n\f$. After
 * discretization, we have
 * \f$
 * \partial_t \rho_n + \mathcal M (\rho_n, \mathbf q_n, \hat \rho_n) = F_n
 * \f$.
 * With
 * \f$
 * \mathcal M (\rho_n, \mathbf q_n, \hat \rho_n)
 * = B_1^T \mathbf q_{nh} + D_1 \rho_{nh} - c_n D_2 \rho_{nh} - E_1 \hat
 * \rho_{nh} + c_n E_2 \hat \rho_{nh}
 * \f$. (with all the matrices defined in reactive_interface::hdg_manager).
 * Then we apply the time stepping scheme:
 * \f[
 * \begin{gathered}
 * \frac 1 {\Delta t} A_1 \rho_{nh}
 * + \mathcal M (\rho_n, \mathbf q_n, \hat \rho_n) =
 * \frac 1 {\Delta t} A_1 \rho_{(n-1),h} +  F_n
 * \end{gathered}
 * \f]
 *
 * In the special case of reactive interface, we follow this step by a
 * Newton increment as follows (note that \f$\mathcal M\f$ is linear w.r.t
 * all of its components):
 * \f[
 * \begin{gathered}
 * \frac 1 {\Delta t} A_1 \delta \rho_{nh}
 * + \mathcal M (\delta \rho_n, \delta \mathbf q_n, \delta \hat \rho_n)
 * + \frac 1 {\Delta t} A_1 \rho_{nh}
 * + \mathcal M (\rho_n, \mathbf q_n, \hat \rho_n) - F_n
 * - \frac 1 {\Delta t} A_1 \rho_{(n-1),h} = 0
 * \end{gathered}
 * \f]
 *
 */
template <typename T> struct BDF1_solver : public first_order_ode_solver
{
  /**
   *
   *
   *
   */
  BDF1_solver(const double h);

  /**
   *
   * @brief Destructor guy
   *
   * This destructor needs to be virtual.
   *
   */
  virtual ~BDF1_solver() final {}

  /**
   *
   *
   *
   */
  void scale_dt(const double sc_fac);

  T x0, x;
  double my_h;
};

/**
 *
 * This structure solves the ODE \f$y'(t) = f(t,y)\f$ using the trapezoidal
 * rule. Having \f$y_0\f$ as the solution in the last time step and \f$y_1\f$ as
 * the solution in the current time step, and \f$h\f$ as time step size,
 * we satisfy: \f$y_1 - y_0 = h [f(t_1,y_1)+f(t_0,y_0)]/2\f$.
 *
 * For example, refer to reactive_interface::hdg_manager, where we have
 * the equaiton: \f$\partial_t \rho_n + \nabla \cdot (\mu_n \alpha_n
 * \lambda^{-2} \mathbf E \rho_n) + \nabla \cdot \mathbf q_n = f_n\f$. After
 * discretization, we have
 * \f$
 * \partial_t \rho_n + \mathcal M (\rho_n, \mathbf q_n, \hat \rho_n) = F_n
 * \f$.
 * With
 * \f$
 * \mathcal M (\rho_n, \mathbf q_n, \hat \rho_n)
 * = B_1^T \mathbf q_{nh} + D_1 \rho_{nh} - c_n D_2 \rho_{nh} - E_1 \hat
 * \rho_{nh} + c_n E_2 \hat \rho_{nh}
 * \f$. (with all the matrices defined in reactive_interface::hdg_manager).
 * Then we apply the time stepping scheme:
 * \f[
 * \begin{gathered}
 * \frac 1 {\Delta t} A_1 \rho_{nh}
 * + \frac 1 2 \mathcal M (\rho_n, \mathbf q_n, \hat \rho_n)
 * = \frac 1 {\Delta t} A_1 \rho_{(n-1),h}
 * - \frac 1 2 \mathcal M (\rho_{n-1}, \mathbf q_{n-1}, \hat \rho_{n-1})
 * + \frac 1 2 (F_n + F_{n-1})
 * \end{gathered}
 * \f]
 *
 * In the special case of reactive interface, we follow this step by a
 * Newton increment as follows (note that \f$\mathcal M\f$ is linear w.r.t
 * all of its arguments):
 * \f[
 * \begin{gathered}
 * \frac 1 {\Delta t} A_1 \delta \rho_{nh}
 * + \frac 1 2 \mathcal M (\delta \rho_n, \delta \mathbf q_n, \delta \hat
 *   \rho_n)
 * + \frac 1 {\Delta t} A_1 \rho_{nh}
 * + \frac 1 2 \mathcal M (\rho_n, \mathbf q_n, \hat \rho_n)
 * = \frac 1 {\Delta t} A_1 \rho_{(n-1),h}
 * - \frac 1 2 \mathcal M (\rho_{n-1}, \mathbf q_{n-1}, \hat \rho_{n-1})
 * + \frac 1 2 (F_n+F_{n-1}).
 * \end{gathered}
 * \f]
 *
 */
template <typename T> struct trapezoidal_solver : public first_order_ode_solver
{
  /**
   *
   * The constructor.
   *
   */
  trapezoidal_solver(const double h);

  /**
   *
   * Since the destrcutor of the base class is virtual, we make this destructor
   * virtual as well.
   *
   */
  virtual ~trapezoidal_solver() final {}

  /**
   *
   *
   *
   */
  T compute_next_x(const T &f0, const T &f1);

  T x0, x;
  double my_h;
};
}
}

#include "../../source/ode_solvers/ode_solvers.cpp"

#endif
