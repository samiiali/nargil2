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

//
//
//
//
//

template <int dim, int spacedim>
nargil::solvers::petsc_direct_solver<dim, spacedim>::petsc_direct_solver(
  const int in_solver_props, const dof_counter<dim, spacedim> &in_dof_counter,
  const MPI_Comm &in_comm)
  : my_comm(&in_comm), my_dof_counter(&in_dof_counter)
{
  my_state = solver_state::solver_not_ready;
  int opts0 = solver_update_opts::update_mat | solver_update_opts::update_rhs |
              solver_update_opts::update_sol;
  init_components(in_solver_props, opts0);
}

//
//

template <int dim, int spacedim>
nargil::solvers::petsc_direct_solver<dim, spacedim>::~petsc_direct_solver()
{
  int update_opts = solver_update_opts::update_mat |
                    solver_update_opts::update_rhs |
                    solver_update_opts::update_sol;
  free_components(update_opts);
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_direct_solver<dim, spacedim>::init_components(
  const int in_solver_props, const int update_opts)
{
  assert(my_state == solver_state::solver_not_ready);
  if (update_opts & solver_update_opts::update_mat)
  {
    MatCreate(*my_comm, &A);
    MatSetType(A, MATMPIAIJ);
    MatSetSizes(A,
                my_dof_counter->n_global_unkns_rank_owns,
                my_dof_counter->n_global_unkns_rank_owns,
                my_dof_counter->n_global_unkns_on_all_ranks,
                my_dof_counter->n_global_unkns_on_all_ranks);
    MatMPIAIJSetPreallocation(
      A,
      0,
      my_dof_counter->n_local_unkns_connected_to_unkn.data(),
      0,
      my_dof_counter->n_nonlocal_unkns_connected_to_unkn.data());
    MatSetOption(A, MAT_ROW_ORIENTED, PETSC_FALSE);
    if (in_solver_props & solver_props::spd_matrix)
      MatSetOption(A, MAT_SPD, PETSC_TRUE);
    if (in_solver_props & solver_props::symmetric_matrix)
      MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);
    if (in_solver_props & solver_props::ignore_mat_zero_entries)
      MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
  }
  if (update_opts & solver_update_opts::update_rhs)
  {
    VecCreateMPI(*my_comm,
                 my_dof_counter->n_global_unkns_rank_owns,
                 my_dof_counter->n_global_unkns_on_all_ranks,
                 &b);
    VecSetOption(b, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  }
  if (update_opts & solver_update_opts::update_sol)
  {
    VecCreateMPI(*my_comm,
                 my_dof_counter->n_global_unkns_rank_owns,
                 my_dof_counter->n_global_unkns_on_all_ranks,
                 &exact_sol);
    VecSetOption(exact_sol, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  }
  my_state = solver_state::operators_created;
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_direct_solver<dim, spacedim>::reinit_components(
  const int in_solver_props, const int update_opts)
{
  free_components(update_opts);
  init_components(in_solver_props, update_opts);
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_direct_solver<dim, spacedim>::free_components(
  const int update_opts)
{
  if ((my_state & solver_state::operators_created) ||
      (my_state & solver_state::assemble_is_finished) ||
      (my_state & solver_state::ready_to_solve))
  {
    if (update_opts & solver_update_opts::update_mat)
      MatDestroy(&A);
    if (update_opts & solver_update_opts::update_rhs)
      VecDestroy(&b);
    if (update_opts & solver_update_opts::update_sol)
      VecDestroy(&exact_sol);
  }
  if (my_state & solver_state::ready_to_solve)
  {
    KSPDestroy(&ksp);
  }
  my_state = solver_state::solver_not_ready;
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_direct_solver<dim, spacedim>::push_to_global_mat(
  const int *rows, const int *cols, const Eigen::MatrixXd &mat,
  const InsertMode ins_mode)
{
  assert(my_state == solver_state::operators_created);
  unsigned n_rows = mat.rows();
  unsigned n_cols = mat.cols();
  MatSetValues(A, n_rows, rows, n_cols, cols, mat.data(), ins_mode);
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_direct_solver<dim, spacedim>::push_to_rhs_vec(
  const int *rows, const Eigen::VectorXd &vec, const InsertMode ins_mode)
{
  assert(my_state == solver_state::operators_created);
  unsigned n_rows = vec.rows();
  VecSetValues(b, n_rows, rows, vec.data(), ins_mode);
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_direct_solver<dim, spacedim>::push_to_exact_sol(
  const int *rows, const Eigen::VectorXd &vec, const InsertMode ins_mode)
{
  assert(my_state == solver_state::operators_created);
  unsigned n_rows = vec.rows();
  VecSetValues(exact_sol, n_rows, rows, vec.data(), ins_mode);
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_direct_solver<dim, spacedim>::finish_assemble(
  const int keys_)
{
  my_state = solver_state::assemble_is_finished;
  if (keys_ & solver_update_opts::update_mat)
  {
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  }
  if (keys_ & solver_update_opts::update_rhs)
  {
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);
  }
  if (keys_ & solver_update_opts::update_sol)
  {
    VecAssemblyBegin(exact_sol);
    VecAssemblyEnd(exact_sol);
  }
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_direct_solver<dim, spacedim>::form_factors()
{
  assert(my_state == solver_state::assemble_is_finished);
  KSPCreate(*my_comm, &ksp);
  KSPSetOperators(ksp, A, A);
  //
  Mat factor_mat;
  KSPSetType(ksp, KSPPREONLY);
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCLU);
  PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS);
  PCFactorSetUpMatSolverPackage(pc);
  PCFactorGetMatrix(pc, &factor_mat);
  // choosing the parallel computing icntl(28) = 2
  MatMumpsSetIcntl(factor_mat, 28, 2);
  // sequential ordering icntl(7) = 2 */
  MatMumpsSetIcntl(factor_mat, 7, 3);
  // MatMumpsSetIcntl(factor_mat, 29, 2);
  // parallel ordering icntl(29) = 2
  // MatMumpsSetIcntl(factor_mat, 29, 2);
  // threshhold for row pivot detection

  // Iterative refinement //
  MatMumpsSetIcntl(factor_mat, 10, -2);
  // MatMumpsSetIcntl(factor_mat, 11, 1);
  // MatMumpsSetIcntl(factor_mat, 12, 1);

  // Null pivot rows detection //
  MatMumpsSetIcntl(factor_mat, 24, 1);

  // Increase in the estimated memory
  MatMumpsSetIcntl(factor_mat, 14, 500);

  // Numerical pivoting
  MatMumpsSetCntl(factor_mat, 1, 0.1);
  MatMumpsSetCntl(factor_mat, 2, 1.E-14);

  // Null pivot row detection
  MatMumpsSetCntl(factor_mat, 3, -1.E-14);

  // Static pivoting
  MatMumpsSetCntl(factor_mat, 4, 1.E-6);

  // MatMumpsSetCntl(factor_mat, 5, 1.E20);

  //
  //
  //
  my_state = solver_state::ready_to_solve;
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_direct_solver<dim, spacedim>::solve_system(
  Vec *sol_vec)
{
  int my_rank;
  MPI_Comm_rank(*my_comm, &my_rank);
  assert(my_state == solver_state::ready_to_solve);
  KSPConvergedReason how_ksp_stopped;
  PetscInt num_iter;
  VecDuplicate(b, sol_vec);
  KSPSolve(ksp, b, *sol_vec);
  KSPGetIterationNumber(ksp, &num_iter);
  KSPGetConvergedReason(ksp, &how_ksp_stopped);
  if (my_rank == 0)
    std::cout << num_iter << "  " << how_ksp_stopped << std::endl;
}

//
//

template <int dim, int spacedim>
std::vector<double> nargil::solvers::petsc_direct_solver<
  dim, spacedim>::get_local_part_of_global_vec(Vec *petsc_vec,
                                               const bool destroy_petsc_vec)
{
  IS from, to;
  Vec local_petsc_vec;
  VecScatter scatter;
  VecCreateSeq(PETSC_COMM_SELF, my_dof_counter->n_local_unkns_on_this_rank,
               &local_petsc_vec);
  ISCreateGeneral(PETSC_COMM_SELF,
                  my_dof_counter->n_local_unkns_on_this_rank,
                  my_dof_counter->scatter_from.data(),
                  PETSC_COPY_VALUES,
                  &from);
  ISCreateGeneral(PETSC_COMM_SELF,
                  my_dof_counter->n_local_unkns_on_this_rank,
                  my_dof_counter->scatter_to.data(),
                  PETSC_COPY_VALUES,
                  &to);
  VecScatterCreate(*petsc_vec, from, local_petsc_vec, to, &scatter);
  VecScatterBegin(scatter, *petsc_vec, local_petsc_vec, INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(scatter, *petsc_vec, local_petsc_vec, INSERT_VALUES,
                SCATTER_FORWARD);
  double *local_exact_pointer;
  VecGetArray(local_petsc_vec, &local_exact_pointer);
  std::vector<double> local_vec(local_exact_pointer,
                                local_exact_pointer +
                                  my_dof_counter->n_local_unkns_on_this_rank);
  {
    VecRestoreArray(local_petsc_vec, &local_exact_pointer);
    VecDestroy(&local_petsc_vec);
    ISDestroy(&from);
    ISDestroy(&to);
    VecScatterDestroy(&scatter);
    if (destroy_petsc_vec)
      VecDestroy(petsc_vec);
  }
  return local_vec;
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::solvers::petsc_implicit_cg_solver<dim, spacedim>::
  petsc_implicit_cg_solver(const int in_solver_props,
                           const dof_counter<dim, spacedim> &in_dof_counter,
                           const MPI_Comm &in_comm)
  : my_comm(&in_comm), my_dof_counter(&in_dof_counter)
{
  my_state = solver_state::solver_not_ready;
  int opts0 = solver_update_opts::update_mat | solver_update_opts::update_rhs |
              solver_update_opts::update_sol;
  init_components(in_solver_props, opts0);
}

//
//

template <int dim, int spacedim>
nargil::solvers::petsc_implicit_cg_solver<dim,
                                          spacedim>::~petsc_implicit_cg_solver()
{
  int update_opts = solver_update_opts::update_mat |
                    solver_update_opts::update_rhs |
                    solver_update_opts::update_sol;
  free_components(update_opts);
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_implicit_cg_solver<dim, spacedim>::init_components(
  const int in_solver_props, const int update_opts)
{
  assert(my_state == solver_state::solver_not_ready);
  if (update_opts & solver_update_opts::update_mat)
  {
    MatCreate(*my_comm, &A);
    MatSetType(A, MATMPIAIJ);
    MatSetSizes(A,
                my_dof_counter->n_global_unkns_rank_owns,
                my_dof_counter->n_global_unkns_rank_owns,
                my_dof_counter->n_global_unkns_on_all_ranks,
                my_dof_counter->n_global_unkns_on_all_ranks);
    MatMPIAIJSetPreallocation(
      A,
      0,
      my_dof_counter->n_local_unkns_connected_to_unkn.data(),
      0,
      my_dof_counter->n_nonlocal_unkns_connected_to_unkn.data());
    MatSetOption(A, MAT_ROW_ORIENTED, PETSC_FALSE);
    if (in_solver_props & solver_props::spd_matrix)
      MatSetOption(A, MAT_SPD, PETSC_TRUE);
    if (in_solver_props & solver_props::symmetric_matrix)
      MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);
    if (in_solver_props & solver_props::ignore_mat_zero_entries)
      MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
  }
  if (update_opts & solver_update_opts::update_rhs)
  {
    VecCreateMPI(*my_comm,
                 my_dof_counter->n_global_unkns_rank_owns,
                 my_dof_counter->n_global_unkns_on_all_ranks,
                 &b);
    VecSetOption(b, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  }
  if (update_opts & solver_update_opts::update_sol)
  {
    VecCreateMPI(*my_comm,
                 my_dof_counter->n_global_unkns_rank_owns,
                 my_dof_counter->n_global_unkns_on_all_ranks,
                 &exact_sol);
    VecSetOption(exact_sol, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  }
  my_state = solver_state::operators_created;
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_implicit_cg_solver<
  dim, spacedim>::reinit_components(const int in_solver_props,
                                    const int update_opts)
{
  free_components(update_opts);
  init_components(in_solver_props, update_opts);
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_implicit_cg_solver<dim, spacedim>::free_components(
  const int update_opts)
{
  if ((my_state & solver_state::operators_created) ||
      (my_state & solver_state::assemble_is_finished) ||
      (my_state & solver_state::ready_to_solve))
  {
    if (update_opts & solver_update_opts::update_mat)
      MatDestroy(&A);
    if (update_opts & solver_update_opts::update_rhs)
      VecDestroy(&b);
    if (update_opts & solver_update_opts::update_sol)
      VecDestroy(&exact_sol);
  }
  if (my_state & solver_state::ready_to_solve)
  {
    KSPDestroy(&ksp);
  }
  my_state = solver_state::solver_not_ready;
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_implicit_cg_solver<
  dim, spacedim>::push_to_global_mat(const int *rows, const int *cols,
                                     const Eigen::MatrixXd &mat,
                                     const InsertMode ins_mode)
{
  assert(my_state == solver_state::operators_created);
  unsigned n_rows = mat.rows();
  unsigned n_cols = mat.cols();
  MatSetValues(A, n_rows, rows, n_cols, cols, mat.data(), ins_mode);
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_implicit_cg_solver<dim, spacedim>::push_to_rhs_vec(
  const int *rows, const Eigen::VectorXd &vec, const InsertMode ins_mode)
{
  assert(my_state == solver_state::operators_created);
  unsigned n_rows = vec.rows();
  VecSetValues(b, n_rows, rows, vec.data(), ins_mode);
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_implicit_cg_solver<
  dim, spacedim>::push_to_exact_sol(const int *rows, const Eigen::VectorXd &vec,
                                    const InsertMode ins_mode)
{
  assert(my_state == solver_state::operators_created);
  unsigned n_rows = vec.rows();
  VecSetValues(exact_sol, n_rows, rows, vec.data(), ins_mode);
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_implicit_cg_solver<dim, spacedim>::finish_assemble(
  const int keys_)
{
  my_state = solver_state::assemble_is_finished;
  if (keys_ & solver_update_opts::update_mat)
  {
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  }
  if (keys_ & solver_update_opts::update_rhs)
  {
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);
  }
  if (keys_ & solver_update_opts::update_sol)
  {
    VecAssemblyBegin(exact_sol);
    VecAssemblyEnd(exact_sol);
  }
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_implicit_cg_solver<dim, spacedim>::form_factors()
{
  assert(my_state == solver_state::assemble_is_finished);
  KSPCreate(*my_comm, &ksp);
  KSPSetOperators(ksp, A, A);
  //
  Mat factor_mat;
  KSPSetType(ksp, KSPCG);
  KSPSetFromOptions(ksp);
  KSPSetTolerances(ksp, 1.E-6, PETSC_DEFAULT, PETSC_DEFAULT, 40000);
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCHYPRE);
  PCSetFromOptions(pc);
  // PCGAMGSetNSmooths(pc, 1);
  //
  my_state = solver_state::ready_to_solve;
}

//
//

template <int dim, int spacedim>
void nargil::solvers::petsc_implicit_cg_solver<dim, spacedim>::solve_system(
  Vec *sol_vec)
{
  int my_rank;
  MPI_Comm_rank(*my_comm, &my_rank);
  assert(my_state == solver_state::ready_to_solve);
  KSPConvergedReason how_ksp_stopped;
  PetscInt num_iter;
  VecDuplicate(b, sol_vec);
  KSPSolve(ksp, b, *sol_vec);
  KSPGetIterationNumber(ksp, &num_iter);
  KSPGetConvergedReason(ksp, &how_ksp_stopped);
  if (my_rank == 0)
    std::cout << num_iter << "  " << how_ksp_stopped << std::endl;
}

//
//

template <int dim, int spacedim>
std::vector<double> nargil::solvers::petsc_implicit_cg_solver<
  dim, spacedim>::get_local_part_of_global_vec(Vec *petsc_vec,
                                               const bool destroy_petsc_vec)
{
  IS from, to;
  Vec local_petsc_vec;
  VecScatter scatter;
  VecCreateSeq(PETSC_COMM_SELF, my_dof_counter->n_local_unkns_on_this_rank,
               &local_petsc_vec);
  ISCreateGeneral(PETSC_COMM_SELF,
                  my_dof_counter->n_local_unkns_on_this_rank,
                  my_dof_counter->scatter_from.data(),
                  PETSC_COPY_VALUES,
                  &from);
  ISCreateGeneral(PETSC_COMM_SELF,
                  my_dof_counter->n_local_unkns_on_this_rank,
                  my_dof_counter->scatter_to.data(),
                  PETSC_COPY_VALUES,
                  &to);
  VecScatterCreate(*petsc_vec, from, local_petsc_vec, to, &scatter);
  VecScatterBegin(scatter, *petsc_vec, local_petsc_vec, INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(scatter, *petsc_vec, local_petsc_vec, INSERT_VALUES,
                SCATTER_FORWARD);
  double *local_exact_pointer;
  VecGetArray(local_petsc_vec, &local_exact_pointer);
  std::vector<double> local_vec(local_exact_pointer,
                                local_exact_pointer +
                                  my_dof_counter->n_local_unkns_on_this_rank);
  {
    VecRestoreArray(local_petsc_vec, &local_exact_pointer);
    VecDestroy(&local_petsc_vec);
    ISDestroy(&from);
    ISDestroy(&to);
    VecScatterDestroy(&scatter);
    if (destroy_petsc_vec)
      VecDestroy(petsc_vec);
  }
  return local_vec;
}
