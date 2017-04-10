#include "petsc.h"

#include "Eigen/Dense"
#include "Eigen/Sparse"

#ifndef SOLVERS_HPP
#define SOLVERS_HPP

namespace nargil
{

// Forward decleration of dof_counter
template <int dim, int spacedim> struct dof_counter;

/**
 *
 *
 *
 */
namespace solvers
{
/**
 *
 * The namespace containing the flags.
 *
 */
namespace solver_update_opts
{
/**
 *
 * @brief The opts enum
 *
 */
enum opts
{
  update_mat = 1 << 0,
  update_rhs = 1 << 1,
  update_sol = 1 << 2
};
}

/**
 *
 * The namespace containing how we insert values to global matrices.
 *
 */
namespace insert_mode
{
/**
 *
 * @brief The mode enum
 *
 */
enum mode
{
  add_vals = 1 << 0,
  ins_vals = 1 << 1
};
}

/**
 *
 * The namespace of the enum for the factorization type.
 *
 */
namespace factorization_type
{
/**
 *
 * @brief The type enum
 *
 */
enum type
{
  lu = 1 << 0
};
}

/**
 *
 * Properties of the solver, such as matrix being spd or symmetric.
 *
 */
namespace solver_props
{
enum props
{
  default_option = 0,
  spd_matrix = 1 << 0,
  symmetric_matrix = 1 << 1,
  ignore_mat_zero_entries = 1 << 2
};
}

/**
 *
 * The namespace for the state of the solver.
 *
 */
namespace solver_state
{
/**
 *
 * @brief The state enum
 *
 */
enum state
{
  solver_not_ready = 1 << 0,
  operators_created = 1 << 1,
  assemble_is_finished = 1 << 2,
  ready_to_solve = 1 << 3
};
}

/**
 *
 *
 * The base class for all implicit solvers.
 *
 *
 */
template <int dim, int spacedim> struct base_implicit_solver
{
  /**
   *
   * The constructor.
   *
   */
  base_implicit_solver();

  /**
   *
   * The destructor.
   *
   */
  virtual ~base_implicit_solver() {}

  /**
   *
   * @brief initializes the matrix and rhs vec and exact_sol vec.
   *
   */
  virtual void push_to_global_mat(const int *rows, const int *cols,
                                  const Eigen::MatrixXd &vals,
                                  const InsertMode ins_mode) = 0;

  /**
   *
   * @brief initializes the matrix and rhs vec and exact_sol vec.
   *
   */
  virtual void push_to_rhs_vec(const int *rows, const Eigen::VectorXd &vals,
                               const InsertMode ins_mode) = 0;

  /**
   *
   * @brief push values to exact solution vector x_exact
   *
   */
  virtual void push_to_exact_sol(const int *rows, const Eigen::VectorXd &vals,
                                 const InsertMode ins_mode) = 0;
};

/**
 *
 *
 * This is the simplest implicit solver in our code. It uses a full
 * matrix from the Eigen3 numerical library. We will implement another
 * more advanced solver with Eigen3 sparse matrix.
 *
 *
 */
template <int dim, int spacedim = dim>
struct simple_implicit_solver : public base_implicit_solver<dim, spacedim>
{
  /**
   *
   * @brief simple_implicit_solver
   *
   */
  simple_implicit_solver(const dof_counter<dim, spacedim> &);

  /**
   *
   * @brief initializes the matrix and rhs vec and exact_sol vec.
   *
   */
  void init_components(const int update_opts);

  /**
   *
   * @brief reinitializes the matrix and rhs vec and exact_sol vec.
   *
   */
  void reinit_components(const int update_opts);

  /**
   *
   * @brief frees the memory of the matrix and rhs vec and exact_sol vec.
   *
   */
  void free_components(const int update_opts);

  /**
   *
   * @brief initializes the matrix and rhs vec and exact_sol vec.
   *
   */
  void push_to_global_mat(const int *rows, const int *cols,
                          const Eigen::MatrixXd &vals,
                          const InsertMode ins_mode);

  /**
   *
   * @brief initializes the matrix and rhs vec and exact_sol vec.
   *
   */
  void push_to_rhs_vec(const int *rows, const Eigen::VectorXd &vals,
                       const InsertMode ins_mode);

  /**
   *
   * @brief push values to exact solution vector x_exact
   *
   */
  void push_to_exact_sol(const int *rows, const Eigen::VectorXd &vals,
                         const InsertMode ins_mode);

  /**
   *
   * @brief Finishes the assembling process.
   *
   */
  void finish_assemble();

  /**
   *
   * @brief factors the system
   *
   */
  void form_factors();

  /**
   *
   * @brief solves the system
   *
   */
  void solve_system(Eigen::VectorXd &sol);

  /**
   *
   * @brief my_dof_counter
   *
   */
  const dof_counter<dim, spacedim> *my_dof_counter;

  /**
   *
   * The sparse global stiffness matrix.
   *
   */
  Eigen::SparseMatrix<double> A;

  /**
   *
   * @brief b
   *
   */
  Eigen::VectorXd b;

  /**
   *
   * @brief x
   *
   */
  Eigen::VectorXd exact_sol;

  /**
   *
   * The lu decomposition of the global matrix.
   *
   */
  Eigen::SparseLU<Eigen::SparseMatrix<double> > lu_of_A;

  /**
   *
   * @brief the current state of the solver
   *
   */
  solver_state::state my_state;
};

//
//
//
//
//

template <int dim, int spacedim = dim>
struct petsc_direct_solver : public base_implicit_solver<dim, spacedim>
{
  /**
   *
   * @brief The constructor of the class.
   *
   */
  petsc_direct_solver(const int in_solver_props,
                      const dof_counter<dim, spacedim> &,
                      const MPI_Comm &in_comm);

  /**
   *
   *
   *
   */
  virtual ~petsc_direct_solver();

  /**
   *
   * @brief initializes the matrix and rhs vec and exact_sol vec.
   *
   */
  void init_components(const int in_solver_props, const int update_opts);

  /**
   *
   * @brief reinitializes the matrix and rhs vec and exact_sol vec.
   *
   */
  void reinit_components(const int in_solver_props, const int update_opts);

  /**
   *
   * @brief frees the memory of the matrix and rhs vec and exact_sol vec.
   *
   */
  void free_components(const int update_opts);

  /**
   *
   * @brief initializes the matrix and rhs vec and exact_sol vec.
   *
   */
  void push_to_global_mat(const int *rows, const int *cols,
                          const Eigen::MatrixXd &vals,
                          const InsertMode ins_mode);

  /**
   *
   * @brief initializes the matrix and rhs vec and exact_sol vec.
   *
   */
  void push_to_rhs_vec(const int *rows, const Eigen::VectorXd &vals,
                       const InsertMode ins_mode);

  /**
   *
   * @brief push values to exact solution vector x_exact
   *
   */
  void push_to_exact_sol(const int *rows, const Eigen::VectorXd &vals,
                         const InsertMode ins_mode);

  /**
   *
   * @brief Finishes the assembling process.
   *
   */
  void finish_assemble(const int keys);

  /**
   *
   * @brief factors the system
   *
   */
  void form_factors();

  /**
   *
   * @brief solves the system
   *
   */
  void solve_system(Vec *sol);

  /**
   *
   *
   *
   */
  std::vector<double>
  get_local_part_of_global_vec(Vec *petsc_vec,
                               const bool destroy_petsc_vec = true);

  /**
   *
   *
   *
   */
  const MPI_Comm *my_comm;

  /**
   *
   * @brief my_dof_counter
   *
   */
  const dof_counter<dim, spacedim> *my_dof_counter;

  /**
   *
   * The sparse global stiffness matrix.
   *
   */
  Mat A;

  /**
   *
   * @brief b
   *
   */
  Vec b;

  /**
   *
   * @brief x
   *
   */
  Vec exact_sol;

  /**
   *
   *
   *
   */
  PC pc;

  /**
   *
   * the krylov space of the solver.
   *
   */
  KSP ksp;

  /**
   *
   * @brief the current state of the solver
   *
   */
  solver_state::state my_state;
};

//
//
//
//
//

template <int dim, int spacedim = dim>
struct petsc_implicit_cg_solver : public base_implicit_solver<dim, spacedim>
{
  /**
   *
   * @brief The constructor of the class.
   *
   */
  petsc_implicit_cg_solver(const int in_solver_props,
                           const dof_counter<dim, spacedim> &,
                           const MPI_Comm &in_comm);

  /**
   *
   *
   *
   */
  virtual ~petsc_implicit_cg_solver();

  /**
   *
   * @brief initializes the matrix and rhs vec and exact_sol vec.
   *
   */
  void init_components(const int in_solver_props, const int update_opts);

  /**
   *
   * @brief reinitializes the matrix and rhs vec and exact_sol vec.
   *
   */
  void reinit_components(const int in_solver_props, const int update_opts);

  /**
   *
   * @brief frees the memory of the matrix and rhs vec and exact_sol vec.
   *
   */
  void free_components(const int update_opts);

  /**
   *
   * @brief initializes the matrix and rhs vec and exact_sol vec.
   *
   */
  void push_to_global_mat(const int *rows, const int *cols,
                          const Eigen::MatrixXd &vals,
                          const InsertMode ins_mode);

  /**
   *
   * @brief initializes the matrix and rhs vec and exact_sol vec.
   *
   */
  void push_to_rhs_vec(const int *rows, const Eigen::VectorXd &vals,
                       const InsertMode ins_mode);

  /**
   *
   * @brief push values to exact solution vector x_exact
   *
   */
  void push_to_exact_sol(const int *rows, const Eigen::VectorXd &vals,
                         const InsertMode ins_mode);

  /**
   *
   * @brief Finishes the assembling process.
   *
   */
  void finish_assemble(const int keys);

  /**
   *
   * @brief factors the system
   *
   */
  void form_factors();

  /**
   *
   * @brief solves the system
   *
   */
  void solve_system(Vec *sol);

  /**
   *
   *
   *
   */
  std::vector<double>
  get_local_part_of_global_vec(Vec *petsc_vec,
                               const bool destroy_petsc_vec = true);

  /**
   *
   *
   *
   */
  const MPI_Comm *my_comm;

  /**
   *
   * @brief my_dof_counter
   *
   */
  const dof_counter<dim, spacedim> *my_dof_counter;

  /**
   *
   * The sparse global stiffness matrix.
   *
   */
  Mat A;

  /**
   *
   * @brief b
   *
   */
  Vec b;

  /**
   *
   * @brief x
   *
   */
  Vec exact_sol;

  /**
   *
   *
   *
   */
  PC pc;

  /**
   *
   * the krylov space of the solver.
   *
   */
  KSP ksp;

  /**
   *
   * @brief the current state of the solver
   *
   */
  solver_state::state my_state;
};
}
}

#include "../../source/solvers/solvers.cpp"

#endif
