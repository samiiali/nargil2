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
 *
 */
struct first_order_ode_solver
{
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
 * discretization, we have the following matrix
 * form for this (with all the matrices defined in
 * reactive_interface::hdg_manager):
 * \f[
 * \begin{gathered}
 * h_t A_1 \delta \rho_{nh} +
 * B_1^T \delta \mathbf q_{nh} + D_1 \delta \rho_{nh} - c_n D_2 \delta \rho_{nh}
 * - E_1 \delta \hat \rho_{nh} + c_n E_2 \delta \hat \rho_{nh}
 * + h_t A_1 \rho_{nh} +
 * B_1^T \mathbf q_{nh} + D_1 \rho_{nh} - c_n D_2 \rho_{nh} - E_1 \hat
 * \rho_{nh} + c_n E_2
 * \hat \rho_{nh} - F_n - h_t A_1 \rho^0_{nh} = 0
 * \end{gathered}
 * \f]
 * Obviously, we have first applied time stepping and then the NR
 * iteration increments.
 *
 */
template <typename T> struct BDF1_solver : public first_order_ode_solver
{
  T x0, x;
};
}
}

#include "../../source/ode_solvers/ode_solvers.cpp"

#endif
