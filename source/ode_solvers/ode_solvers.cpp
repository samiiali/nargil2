#include "../../include/ode_solvers/ode_solver.hpp"

nargil::ode_solvers::trapezoidal_solver::trapezoidal_solver(const double h)
  : my_h(h)
{
}

//
//

T nargil::ode_solvers::trapezoidal_solver::trapezoidal_solver::compute_next_x(
  const T &f0, const T &f1)
{
  x = x0 + h / 2. * (f0 + f1);
}
