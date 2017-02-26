#include "../../include/solvers/solvers.hpp"

template <int dim, int spacedim>
nargil::solvers::base_implicit_solver<dim, spacedim>::base_implicit_solver()
{
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::solvers::simple_implicit_solver<dim, spacedim>::simple_implicit_solver(
  const dof_counter<dim, spacedim> &in_dof_counter)
  : my_dof_counter(&in_dof_counter)
{
  my_state = solver_state::solver_not_ready;
  int opts0 = solver_update_opts::update_mat | solver_update_opts::update_rhs |
              solver_update_opts::update_sol;
  init_components(opts0);
}

//
//

template <int dim, int spacedim>
void nargil::solvers::simple_implicit_solver<dim, spacedim>::init_components(
  const int update_opts)
{
  assert(my_state == solver_state::solver_not_ready);
  unsigned n_unkns = my_dof_counter->n_global_unkns_on_all_ranks;
  if (update_opts & solver_update_opts::update_mat)
  {
    A.resize(n_unkns, n_unkns);
    A.reserve(my_dof_counter->n_local_unkns_connected_to_unkn);
    //    Sp_A.setZero();
  }
  if (update_opts & solver_update_opts::update_rhs)
  {
    b = std::move(Eigen::VectorXd::Zero(n_unkns));
  }
  if (update_opts & solver_update_opts::update_sol)
  {
    exact_sol = std::move(Eigen::VectorXd::Zero(n_unkns));
  }
  my_state = solver_state::operators_created;
}

//
//

template <int dim, int spacedim>
void nargil::solvers::simple_implicit_solver<dim, spacedim>::reinit_components(
  const int update_opts)
{
  my_state = solver_state::solver_not_ready;
  init_components(update_opts);
}

//
//

template <int dim, int spacedim>
void nargil::solvers::simple_implicit_solver<dim, spacedim>::free_components(
  const int update_opts)
{
  if (update_opts & solver_update_opts::update_mat)
  {
    A.resize(0, 0);
    A.data().squeeze();
  }
  if (update_opts & solver_update_opts::update_rhs)
  {
    b.resize(0);
  }
  if (update_opts & solver_update_opts::update_sol)
  {
    exact_sol.resize(0);
  }
  my_state = solver_state::solver_not_ready;
}

//
//

template <int dim, int spacedim>
void nargil::solvers::simple_implicit_solver<dim, spacedim>::push_to_global_mat(
  const int *rows, const int *cols, const Eigen::MatrixXd &mat,
  const InsertMode ins_mode)
{
  assert(my_state == solver_state::operators_created);
  unsigned n_rows = mat.rows();
  unsigned n_cols = mat.cols();
  for (unsigned i_col = 0; i_col < n_cols; ++i_col)
  {
    for (unsigned i_row = 0; i_row < n_rows; ++i_row)
    {
      if (rows[i_row] >= 0 && cols[i_col] >= 0)
      {
        if (ins_mode == INSERT_VALUES)
        {
          A.insert(rows[i_row], cols[i_col]) = mat(i_row, i_col);
        }
        if (ins_mode & ADD_VALUES)
        {
          A.coeffRef(rows[i_row], cols[i_col]) += mat(i_row, i_col);
        }
      }
    }
  }
}

//
//

template <int dim, int spacedim>
void nargil::solvers::simple_implicit_solver<dim, spacedim>::push_to_rhs_vec(
  const int *rows, const Eigen::VectorXd &vec, const InsertMode ins_mode)
{
  assert(my_state == solver_state::operators_created);
  unsigned n_rows = vec.rows();
  for (unsigned i_row = 0; i_row < n_rows; ++i_row)
  {
    if (rows[i_row] >= 0)
    {
      if (ins_mode & INSERT_VALUES)
        b(rows[i_row]) = vec(i_row);
      if (ins_mode & ADD_VALUES)
        b(rows[i_row]) += vec(i_row);
    }
  }
}

//
//

template <int dim, int spacedim>
void nargil::solvers::simple_implicit_solver<dim, spacedim>::push_to_exact_sol(
  const int *rows, const Eigen::VectorXd &vec, const InsertMode ins_mode)
{
  assert(my_state == solver_state::operators_created);
  unsigned n_rows = vec.rows();
  for (unsigned i_row = 0; i_row < n_rows; ++i_row)
  {
    if (rows[i_row] >= 0)
    {
      if (ins_mode & INSERT_VALUES)
        exact_sol(rows[i_row]) = vec(i_row);
      if (ins_mode & ADD_VALUES)
        exact_sol(rows[i_row]) += vec(i_row);
    }
  }
}

//
//

template <int dim, int spacedim>
void nargil::solvers::simple_implicit_solver<dim, spacedim>::finish_assemble()
{
  A.makeCompressed();
  my_state = solver_state::assemble_is_finished;
}

//
//

template <int dim, int spacedim>
void nargil::solvers::simple_implicit_solver<dim, spacedim>::form_factors()
{
  assert(my_state == solver_state::assemble_is_finished);
  lu_of_A.analyzePattern(A);
  lu_of_A.factorize(A);
  my_state = solver_state::ready_to_solve;
}

//
//

template <int dim, int spacedim>
void nargil::solvers::simple_implicit_solver<dim, spacedim>::solve_system(
  Eigen::VectorXd &sol)
{
  assert(my_state == solver_state::ready_to_solve);
  sol = lu_of_A.solve(b);
}
