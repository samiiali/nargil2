#include "../../include/ode_solvers/ode_solver.hpp"

template <typename T>
nargil::ode_solvers::trapezoidal_solver<T>::trapezoidal_solver(const double h)
  : my_h(h)
{
}

//
//

template <typename T>
T nargil::ode_solvers::trapezoidal_solver<T>::compute_next_x(const T &f0,
                                                             const T &f1)
{
  x = x0 + my_h / 2. * (f0 + f1);
}
