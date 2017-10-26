#include "../../include/ode_solvers/ode_solver.hpp"

template <typename T>
nargil::ode_solvers::BDF1_solver<T>::BDF1_solver(const double h) : my_h(h)
{
}

//
//

template <typename T>
void nargil::ode_solvers::BDF1_solver<T>::scale_dt(const double sc_fac)
{
  assert(sc_fac > 1.e-12);
  my_h *= sc_fac;
}

//
//
//
//
//

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
