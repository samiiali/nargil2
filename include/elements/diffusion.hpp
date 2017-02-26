#include <type_traits>

#include <boost/dynamic_bitset.hpp>

#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/numerics/data_out.h>

#include <Eigen/Dense>

#include "../misc/utils.hpp"
#include "../models/model_options.hpp"
#include "../solvers/solvers.hpp"
#include "cell.hpp"

#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP

namespace nargil
{
/**
 *
 *
 * The element to be used for solving the diffusion equation.
 * The first method of solving this is based on hybridized DG.
 *
 * \ingroup modelelements
 *
 *
 */
template <int dim, int spacedim = dim>
struct diffusion : public cell<dim, spacedim>
{
  /**
   *
   * We use the same typename as we defined in the base class
   * (nargil::cell).
   *
   */
  using typename cell<dim, spacedim>::dealiiTriCell;

  /**
   *
   * The constructor of the class.
   *
   */
  diffusion(dealiiTriCell *in_cell,
            const unsigned id_num_,
            base_basis<dim, spacedim> *base_basis);

  /**
   *
   * The destructor of the class.
   *
   */
  virtual ~diffusion() final;

  /**
   *
   * This function is called from cell::create. This function cannot
   * be const, because the diffusion::my_manager is changed in this
   * function.
   *
   */
  template <typename CellManagerType, typename BasisType>
  void init_manager(const BasisType *basis);

  /**
   *
   * Assigns the boundary condition to different DOFs.
   *
   */
  template <typename BasisType, typename Func> void assign_BCs(Func f);

  /**
   *
   * This function static_cast the manager of this cell to an appropriate
   * type and return it to the user.
   *
   */
  template <typename CellManagerType> CellManagerType *get_manager();

  /**
   *
   * This function static_cast the basis of this cell to an appropriate
   * type and return it to the user.
   *
   */
  template <typename BasisType> const BasisType *get_basis() const;

  /**
   *
   * Contains the basis of the current cell.
   *
   */
  base_basis<dim, spacedim> *my_basis;

  /**
   *
   * The manager of this element. The reason to define this manager
   * as a unique_ptr is we need its polymorphic features in the code.
   * Hence, we are able to use static_cast to cast this manager to derived
   * types of cell_manager.
   *
   */
  std::unique_ptr<cell_manager<dim, spacedim> > my_manager;

  /**
   *
   *
   * Polynomial basis of type HDG.
   *
   * @ingroup modelbases
   *
   *
   */
  struct hdg_polybasis : public base_basis<dim, spacedim>
  {
    /**
     *
     *  CellManagerType
     *
     */
    typedef typename diffusion::template hdg_manager<hdg_polybasis>
      CellManagerType;

    /**
     *
     *  hdg_polybasis
     *
     */
    hdg_polybasis() = delete;

    /**
     *
     * Class constructor
     *
     */
    hdg_polybasis(const unsigned poly_order, const unsigned quad_order);

    /**
     *
     * The destructor of the class.
     *
     */
    virtual ~hdg_polybasis() final;

    /**
     *
     * This function returns the number of dofs on each face of the element.
     * The term dof refers to the functions which we want to solve a global
     * equations to obtain them. For example in diffusion equation we solve
     * for u globally. So, the only global dof is u, and this function
     * returns 1.
     *
     */
    static unsigned get_n_dofs_per_face();

    /**
     *
     *  get_n_unkns_per_dofs
     *
     */
    std::vector<unsigned> get_n_unkns_per_dofs() const;

    /**
     *
     * adjusted_subface_quad_points
     *
     */
    dealii::Point<dim - 1>
    adjusted_subface_quad_points(const dealii::Point<dim - 1> &P0,
                                 const unsigned half_range);

    /**
     *
     * Returns the finite element basis for the local basis of the element.
     *
     */
    const dealii::FESystem<dim> *get_local_fe() const;

    /**
     *
     * Returns the finite element basis for the trace dofs of the element.
     *
     */
    const dealii::FE_FaceQ<dim> *get_trace_fe() const;

    /**
     *
     * local_fe
     *
     */
    dealii::FESystem<dim> local_fe;

    /**
     *
     * trace_fe
     *
     */
    dealii::FE_FaceQ<dim> trace_fe;

    /**
     *
     * fe_vals of local dofs inside the cells.
     *
     */
    std::unique_ptr<dealii::FEValues<dim> > local_fe_val_in_cell;

    /**
     *
     * fe_vals of local dofs inside the cells.
     *
     */
    std::unique_ptr<dealii::FEValues<dim> > local_fe_val_at_cell_supp;

    /**
     *
     * fe_vals of trace dofs on the faces of cell.
     *
     */
    std::vector<std::unique_ptr<dealii::FEFaceValues<dim> > > trace_fe_face_val;

    /**
     *
     * fe_vals of trace dofs on the faces of cell.
     *
     */
    std::unique_ptr<dealii::FEFaceValues<dim> > trace_fe_face_val_at_supp;

    /**
     *
     * fe_vals of local dofs on the cell faces.
     *
     */
    std::vector<std::unique_ptr<dealii::FEValues<dim> > > local_fe_val_on_faces;
  };

  /**
   *
   * This structure is used to solve the diffusion equation by hybridized DG in
   * the domain \f$\Omega \subset \mathbb R^d\f$. The equation reads as:
   * \f[
   * - \nabla \cdot (\kappa \nabla u) = f \qquad \text{in } \Omega,
   * \f]
   * with the boundary conditions:
   * \f[
   * \begin{aligned}
   * u = g_D \qquad &\text{on } {\partial \Omega}_D, \\
   *  (-\kappa \nabla u) \cdot \mathbf n = g_N \qquad &\text{on } {\partial
   *   \Omega}_N.
   * \end{aligned}
   * \f]
   * Next, we introduce \f$\mathbf q = -\kappa \nabla u\f$, and form the system
   * of first-order equations:
   * \f[
   * \begin{aligned}
   * \mathbf q + \kappa \nabla u = 0 \\
   * \quad \nabla \cdot \mathbf q = f
   * \end{aligned} \qquad \text{in } \Omega,
   * \f]
   * with the boundary conditions:
   * \f[
   * \begin{aligned}
   * u = g_D & \qquad \text{on } \partial \Omega_D, \\
   * \mathbf q \cdot \mathbf n = g_N & \qquad \text{on } \partial \Omega_N.
   * \end{aligned}
   * \f]
   * We satisfy this equation in the weak sense:
   * \f[
   * \begin{gathered}
   * (\kappa^{-1} \mathbf q_h , \mathbf v)_K - (u_h,\nabla \cdot \mathbf v)_K
   * +\langle \hat u_h , \mathbf v \cdot \mathbf n \rangle_{\partial K} = 0, \\
   * -(\mathbf q_h,\nabla w)_K +
   * \langle \mathbf q^*_h \cdot \mathbf n, w\rangle_{\partial K} = (f,w)_K,
   * \end{gathered} \qquad \forall K \in \mathcal T_h.
   * \f]
   * with \f$\mathbf q^*_h\f$ being the numerical flux. We define it as
   * \f$\mathbf q^*_h = \mathbf q_h + \tau (u_h-\hat u_h) \mathbf n\f$.
   *
   * Next we build the boundary conditions on \f$u\f$ in the space of \f$\hat
   * u\f$ and solve for \f$\lambda\f$ on the mesh skeleton. By including this in
   * the weak form, we will have:
   * \f[
   * \begin{gathered}
   * (\kappa^{-1} \mathbf q_h , \mathbf v)_K - (u_h,\nabla \cdot \mathbf v)_K
   *   +\langle \lambda_h , \mathbf v \cdot \mathbf n \rangle_{\partial K}
   * =
   * -\langle g_D , \mathbf v \cdot \mathbf n
   * \rangle_{\partial K \cap \partial \Omega_D},
   *   \qquad \forall K \in \mathcal T_h \\
   * -(\mathbf q_h,\nabla w)_K +
   *  \langle \mathbf q_h \cdot \mathbf n, w \rangle_{\partial K} + \langle \tau
   *    (u_h-\lambda_h), w\rangle_{\partial K}
   * =
   * (f,w)_K + \langle \tau g_D , w \rangle_{\partial K \cap \partial \Omega_D}
   *    , \qquad \forall K \in \mathcal T_h \\
   * \sum_{K \in \mathcal T_h}
   * \langle \mathbf q^*_h \cdot \mathbf n, \mu \rangle_{\partial K} =
   * \sum_{K\in \mathcal T_h} \langle g_N , \mu \rangle_{\partial K \cap
   *    \partial \Omega_N},
   * \end{gathered}
   * \f]
   * We write this using the following bilinear and functionals:
   * \f[
   * \begin{gathered}
   * a_K (\mathbf q_h, \mathbf v) - b_K (u_h, \mathbf v) +
   * c_{K} (\lambda_h, \mathbf v) = r_K (\mathbf v),
   *  \qquad \forall K \in \mathcal T_h  \\
   * b^T_K (\mathbf q_h, w) + d_K (u_h, w) + e_{K}
   * (\lambda_h, w) = f_K(w) , \qquad \forall K \in \mathcal T_h \\
   * \sum_{K \in \mathcal T_h}
   * \left[c_K^T (\mathbf q_h, \mu) + e^T_K(u_h, \mu) +
   * h_K (\lambda_h, \mu) \right] = \sum _{K\in \mathcal T_h}
   * l_K(\mu),
   * \end{gathered}
   * \f]
   * where,
   * \f[
   * \begin{gathered}
   * a_K (\mathbf q_h,\mathbf v) = (\kappa^{-1} \mathbf q_h , \mathbf
   * v)_K, \quad
   * b_K (u_h,\mathbf v) = (u_h,\nabla \cdot \mathbf v)_K, \quad
   * c_{K} (\lambda_h, \mathbf v) = \langle \lambda_h ,
   * \mathbf v \cdot \mathbf n \rangle_{\partial K}, \\
   * d_K(u_h,w) = \langle \tau u_h, w\rangle_{\partial K}, \quad
   * e_{K}(\lambda_h , w) = \langle -\tau \lambda_h, w\rangle_{\partial
   * K}, \quad
   * h_K(\lambda_h, \mu) = \langle -\tau \lambda_h,
   * \mu\rangle_{\partial K},\\
   * r_{K} (\mathbf v) = -\langle g_D , \mathbf v \cdot \mathbf n
   * \rangle_{\partial K \cap \partial \Omega_D}, \quad
   * f_K(w) = (f,w)_K + \langle \tau g_D , w \rangle_{\partial K \cap
   * \partial \Omega_D}, \quad
   * l_K(\mu) = \langle g_N , \mu
   * \rangle_{\partial K \cap \partial \Omega_N}.
   * \end{gathered}
   * \f]
   * And the matrix form is:
   * \f[
   * A_K Q_K - B_K U_K + C_K \Lambda_K =
   * R_K, \qquad \forall K \in \mathcal T_h\\
   * B^T_K Q_K + D_K U_K + E_K \Lambda_K =
   * F_K, \qquad \forall K \in \mathcal T_h\\
   * C^T Q + E^T U + H \Lambda = L.
   * \f]
   * We want to derive internal variables in terms of trace unknowns:
   * \f[
   * Q_K = A_K^{-1}\left( R_K + B_K U_K - C_K \Lambda_K \right)
   * \f]
   * and
   * \f[
   * U_K = \left(B^T_K A^{-1}_K B_K + D_K\right)^{-1} \left[ F_K -B_K^T
   * A^{-1}_K R_K + (B^T_K A^{-1}_K C_K + E_K) \Lambda_K \right]
   * \f]
   *
   */
  template <typename BasisType>
  struct hdg_manager : hybridized_cell_manager<dim, spacedim>
  {
    /**
     *
     *
     *
     */
    using typename hybridized_cell_manager<dim, spacedim>::funcType;

    /**
     *
     *
     *
     */
    using typename hybridized_cell_manager<dim, spacedim>::vectorFuncType;

    /**
     *
     * hdg_manager
     *
     */
    hdg_manager(const diffusion<dim, spacedim> *, const BasisType *in_basis);

    /**
     *
     *  Deconstructor of the class
     *
     */
    virtual ~hdg_manager() final;

    /**
     *
     * Sets the boundary conditions of the cell.
     *
     */
    template <typename Func> void assign_BCs(Func f);

    /**
     *
     * This function assigns the local_interior_unkn_idx.
     *
     */
    void assign_local_interior_unkn_id(unsigned *local_num);

    /**
     *
     * Assembles the global matrices.
     *
     */
    void
    assemble_globals(solvers::base_implicit_solver<dim, spacedim> *in_solver);

    /**
     *
     *
     *
     */
    void compute_local_unkns(const double *trace_sol);

    /**
     *
     * Computes the local matrices.
     *
     */
    void compute_matrices();

    /**
     *
     *
     *
     */
    void interpolate_to_trace(funcType func);

    /**
     *
     *
     *
     */
    void interpolate_to_interior(vectorFuncType func);

    /**
     *
     *
     *
     */
    void fill_visualization_vector(distributed_vector<dim, spacedim> *out_vec);

    /**
     *
     *
     *
     */
    void set_source_and_BCs(funcType f, funcType g_D, vectorFuncType g_N);

    /**
     *
     *
     *
     */
    void compute_errors(std::vector<double> *sum_of_L2_errors);

    /**
     *
     * This function is used to interpolate the function f to the trace
     * degrees of freedom of the element.
     *
     */
    static void run_interpolate_to_trace(diffusion *in_cell, funcType f);

    /**
     *
     *
     *
     */
    static void run_interpolate_to_interior(diffusion *in_cell,
                                            vectorFuncType f);

    /**
     *
     *
     *
     */
    static void
    run_fill_visualization_vector(diffusion *in_cell,
                                  distributed_vector<dim, spacedim> *out_vec);

    /**
     *
     * This function, sets source term, Dirichlet and Neumann BC functions.
     *
     */
    static void run_set_source_and_BCs(diffusion *in_cell, funcType f,
                                       funcType g_D, vectorFuncType g_N);

    /**
     *
     * compute_local_unkns
     *
     */
    static void run_compute_local_unkns(diffusion *in_cell,
                                        const double *trace_sol);

    /**
     *
     *
     *
     */
    static void run_assemble_globals(
      diffusion *in_cell,
      solvers::base_implicit_solver<dim, spacedim> *in_solver);

    /**
     *
     * Compute the error of u and q, based on the exact_u and exact_q.
     * As a result the function static set_exact_local_dofs should be called
     * before calling this function.
     *
     */
    static void run_compute_errors(diffusion *in_cell,
                                   std::vector<double> *sum_of_L2_errors);

    /**
     *
     *
     *
     */
    static void
    visualize_results(const dealii::DoFHandler<dim, spacedim> &dof_handler,
                      const LA::MPI::Vector &visual_solu,
                      unsigned const &time_level);

    /**
     *
     *
     *
     */
    const BasisType *my_basis;

    /** @{
     *
     * All of the main local matrices of the element.
     *
     */
    Eigen::MatrixXd A, B, C, D, E, H;
    ///@}

    /** @{
     *
     * @brief All of the element local vectors.
     *
     */
    Eigen::VectorXd R, F, L;
    ///@}

    /** @{
     *
     * @brief The exact solutions on the corresponding nodes.
     *
     */
    Eigen::VectorXd exact_uhat, exact_u, exact_q, uhat_vec, u_vec, q_vec;
    ///@}

    /** @{
     *
     * @brief The source term and applied BCs.
     *
     */
    Eigen::VectorXd gD_vec, f_vec;

    /**
     *
     * @brief The std vector of gN's.
     *
     */
    std::vector<dealii::Tensor<1, dim> > gN_vec;
    ///@}

    /**
     *
     * This gives us the interior dof numbers. dealii gives us the global
     * dof numbers.
     *
     */
    std::vector<int> local_interior_unkn_idx;
  };
};
}

#include "../../source/elements/diffusion.cpp"

#endif
