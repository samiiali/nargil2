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

#include "../mesh/mesh_handler.hpp"
#include "../misc/utils.hpp"
#include "../models/model_options.hpp"
#include "../solvers/solvers.hpp"
#include "cell.hpp"
#include "reactive_interface.hpp"

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
  //  using typename cell<dim, spacedim>::dealiiTriCell;

  /**
   *
   * The constructor of the class.
   *
   */
  diffusion(dealiiTriCell<dim, spacedim> *in_cell,
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
   * The boundary condition for diffusion problem
   *
   */
  enum class boundary_condition
  {
    not_set = 0,
    essential = 1,
    natural = 2,
    periodic = 3,
    tokamak_specific = 4
  };

  /**
   *
   *
   * This structure contains all of the required data, such as rhs, BCs, ...
   * to solve the problem.
   *
   *
   */
  struct data
  {
    /**
     * @brief Constructor.
     */
    data() {}

    /**
     * @brief rhs_func.
     */
    virtual double rhs_func(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief gD_func.
     */
    virtual double gD_func(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief gN_func.
     */
    virtual dealii::Tensor<1, dim> gN_func(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief exact_u.
     */
    virtual double exact_u(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief exact_q.
     */
    virtual dealii::Tensor<1, dim> exact_q(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief the inverse of diffusivity tensor.
     */
    virtual dealii::Tensor<2, dim>
    kappa_inv(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief the stabilization parameter.
     */
    virtual double tau(const dealii::Point<spacedim> &) = 0;
  };

  /**
   *
   *
   * This structure contains all of the data for visualizing the solution.
   *
   *
   */
  struct viz_data
  {
    /**
     * @brief The constructor of the structure.
     */
    viz_data(const MPI_Comm in_comm,
             const dealii::DoFHandler<dim, spacedim> *in_dof_handler,
             const dealii::LinearAlgebraPETSc::MPI::Vector *in_viz_sol,
             const std::string &in_filename, const std::string &in_u_name,
             const std::string &in_q_name);

    /**
     * @brief MPI Communicator.
     */
    const MPI_Comm my_comm;

    /**
     * The corresponding deal.II DoFHandler object.
     */
    const dealii::DoFHandler<dim, spacedim> *my_dof_handler;

    /**
     * The deal.II parallel solution vector.
     */
    const LA::MPI::Vector *my_viz_sol;

    /**
     * The output filename.
     */
    const std::string my_out_filename;

    /**
     * A string containing the name of the \f$u\f$ variable in the formulation.
     * This will be displayed in Paraview (like Head or Temperature).
     */
    const std::string my_u_name;

    /**
     * A string containing the name of the \f$u\f$ variable in the formulation.
     * This will be displayed in Paraview (like Hydraulic flow or Heat flow).
     */
    const std::string my_q_name;

    /**
     * @brief The time step and iteration cycle.
     */
    unsigned time_step, cycle;
  };

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
   * Assign the main data such as BC, rhs_func, ... to the cell.
   *
   */
  static void assign_data(diffusion *in_cell, data *in_data);

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
   * This connects the diffusion cell to the R_I cell.
   *
   */
  template <typename OtherCellEq>
  void connect_to_other_cell(OtherCellEq *in_relevant_R_I_cell);

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
   * The data for this cell.
   *
   */
  data *my_data;

  /**
   *
   *
   *
   */
  nargil::reactive_interface<dim, spacedim> *my_relvevant_R_I_cell;

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
     * hdg_polybasis
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
     * Only used in hybridized_model_manager::form_dof_handlers().
     *
     */
    const dealii::FESystem<dim> *get_local_fe() const;

    /**
     *
     * Returns the finite element basis for the trace dofs of the element.
     * Only used in hybridized_model_manager::form_dof_handlers().
     *
     */
    const dealii::FE_FaceQ<dim> *get_trace_fe() const;

    /**
     *
     * Returns the finite element basis for the trace dofs of the element.
     * Only used in hybridized_model_manager::form_dof_handlers().
     *
     */
    const dealii::FE_DGQ<dim> *get_refn_fe() const;

    /**
     *
     * Returns the finite element basis for the visualization basis of the
     * element. Only used in hybridized_model_manager::form_dof_handlers().
     *
     */
    const dealii::FESystem<dim> *get_viz_fe() const;

    /**
     *
     * The number of the quadrature points of the face.
     *
     */
    unsigned get_face_quad_size() const;

    /**
     *
     * The number of the quadrature points of the cell.
     *
     */
    unsigned get_cell_quad_size() const;

    /**
     *
     * @brief Number of unknowns per each interior local dof.
     *
     */
    unsigned n_unkns_per_local_scalar_dof() const;

    /**
     *
     * @brief Number of total trace unknowns in the cell.
     *
     */
    unsigned n_trace_unkns_per_cell() const;

    /**
     *
     * @brief Number of trace unknowns on the face.
     *
     */
    unsigned n_trace_unkns_per_face() const;

    /**
     *
     * @brief Number of local unknowns in each cell.
     *
     */
    unsigned n_local_unkns_per_cell() const;

  private:
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
     * The dealii FiniteElement which is used for refinement criteria.
     *
     */
    dealii::FE_DGQ<dim> refn_fe;

  public:
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
     * This FE Val is used for integrating over faces of element. Since,
     * we are not certain that face support points (i.e. LGL points) will
     * also be used for integration, we form this FE val.
     *
     */
    std::vector<std::unique_ptr<dealii::FEFaceValues<dim> > > trace_fe_face_val;

    /**
     *
     * This FE Val is used for interpolating gD and gN to support points
     * of the element.
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
   * b^T_K (\mathbf q_h, w) + d_K (u_h, w) - e_{K}
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
   * e_{K}(\lambda_h , w) = \langle \tau \lambda_h, w\rangle_{\partial
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
   * B^T_K Q_K + D_K U_K - E_K \Lambda_K =
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
     * The function type with an hdg_manager as input and void output.
     *
     */
    typedef std::function<void(hdg_manager<BasisType> *)> BC_Func;

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
     * Sets the boundary conditions of the cell. This function is called from
     * static assign_BCs().
     *
     */
    void assign_my_BCs(BC_Func f);

    /**
     *
     * This function assigns the local_interior_unkn_idx.
     *
     */
    virtual void set_local_interior_unkn_id(unsigned *local_num) final;

    /**
     *
     * Assembles the global matrices. It is called from static
     * assemble_globals().
     *
     */
    void assemble_my_globals(
      solvers::base_implicit_solver<dim, spacedim> *in_solver);

    /**
     *
     * Assembles the global matrices. It is called from static
     * assemble_globals().
     *
     */
    void assemble_my_tokamak_globals(
      solvers::base_implicit_solver<dim, spacedim> *in_solver,
      std::function<dealii::Tensor<1, dim>(const dealii::Point<spacedim> &p)>
        b_func);

    /**
     *
     * This function computes the local unknowns from the input trace solution.
     * It should not be called directly, but through the static
     * compute_local_unkns().
     *
     */
    void compute_my_local_unkns(const double *trace_sol);

    /**
     *
     * Computes the local matrices.
     *
     */
    void compute_my_matrices();

    /**
     *
     * Computes the local matrices.
     *
     */
    void compute_tokamak_matrices(
      std::function<dealii::Tensor<1, dim>(const dealii::Point<spacedim> &p)>
        b_func);

    /**
     *
     * Called from static interpolate_to_trace().
     *
     */
    void interpolate_to_my_trace();

    /**
     *
     * Called from static interpolate_to_interior().
     *
     */
    void interpolate_to_my_interior();

    /**
     *
     * Called from static fill_viz_vector().
     *
     */
    void fill_my_viz_vector(distributed_vector<dim, spacedim> *out_vec);

    /**
     *
     * Called from static fill_viz_vector().
     *
     */
    void fill_my_viz_vector_with_grad_u_dot_b(
      distributed_vector<dim, spacedim> *out_vec,
      std::function<dealii::Tensor<1, dim>(const dealii::Point<spacedim> &p)>
        b_func);

    /**
     *
     * Called from static fill_viz_vec_with_exact_sol().
     *
     */
    void
    fill_my_viz_vec_with_exact_sol(distributed_vector<dim, spacedim> *out_vec);

    /**
     *
     * Called from fill_refn_vector().
     *
     */
    void fill_my_refn_vector(distributed_vector<dim, spacedim> *out_vec);

    /**
     *
     * Called from set_source_and_BCs().
     *
     */
    void set_my_source_and_BCs();

    /**
     *
     * Called from apply_R_I_source.
     *
     */
    template <typename RelevantCellManagerType> void apply_my_R_I_source();

    /**
     *
     * Called from compute_errors().
     *
     */
    void compute_my_errors(std::vector<double> *sum_of_L2_errors);

    /**
     *
     * This function gives the computed fluxes in the element and on the
     * boundary of the element.
     *
     */
    void set_flux_vector(double **out_q, double **out_q_flux);

    /**
     *
     * This function is used to interpolate the function f to the trace
     * degrees of freedom of the element.
     *
     */
    static void assign_BCs(diffusion *in_cell, BC_Func f);

    /**
     *
     * This function is used to interpolate the function f to the trace
     * degrees of freedom of the element.
     *
     */
    static void interpolate_to_trace(diffusion *in_cell);

    /**
     *
     * Called from outside of the function to interpolate the input function
     * to the interior of the given cell.
     *
     */
    static void interpolate_to_interior(diffusion *in_cell);

    /**
     *
     * Fills the visualization vector of the element.
     *
     */
    static void fill_viz_vector(diffusion *in_cell,
                                distributed_vector<dim, spacedim> *out_vec);

    /**
     *
     * Fills the visualization vector of the element.
     *
     */
    static void fill_viz_vector_with_grad_u_dot_b(
      diffusion *in_cell, distributed_vector<dim, spacedim> *out_vec,
      std::function<dealii::Tensor<1, dim>(const dealii::Point<spacedim> &p)>
        b_func);

    /**
     *
     * Fills the visualization vector with the exact solution.
     *
     */
    static void
    fill_viz_vec_with_exact_sol(diffusion *in_cell,
                                distributed_vector<dim, spacedim> *out_vec);

    /**
     *
     * Fills the refinement vector of the element.
     *
     */
    static void fill_refn_vector(diffusion *in_cell,
                                 distributed_vector<dim, spacedim> *out_vec);

    /**
     *
     * This function, sets source term, Dirichlet and Neumann BC functions.
     *
     */
    static void set_source_and_BCs(diffusion *in_cell);

    /**
     *
     *
     *
     */
    template <typename RelevantCellManagerType>
    static void apply_R_I_source(diffusion *in_cell);

    /**
     *
     * compute_my_local_unkns
     *
     */
    static void compute_local_unkns(diffusion *in_cell,
                                    const double *trace_sol);

    /**
     *
     * Called from outside of the class to assemble global matrices and
     * rhs vector corresponding to this element.
     *
     */
    static void
    assemble_globals(diffusion *in_cell,
                     solvers::base_implicit_solver<dim, spacedim> *in_solver);

    /**
     *
     * Called from outside of the class to assemble global matrices and
     * rhs vector corresponding to this element.
     *
     */
    static void assemble_tokamak_globals(
      diffusion *in_cell,
      solvers::base_implicit_solver<dim, spacedim> *in_solver,
      std::function<dealii::Tensor<1, dim>(const dealii::Point<spacedim> &p)>
        b_func);

    /**
     *
     *
     *
     */
    static void compute_matrices(diffusion *in_cell);

    /**
     *
     * Compute the error of u and q, based on the exact_u and exact_q.
     * As a result the function static set_exact_local_dofs should be called
     * before calling this function.
     *
     */
    static void compute_errors(diffusion *in_cell,
                               std::vector<double> *sum_of_L2_errors);

    /**
     *
     * This function writes the vtk output.
     *
     */
    static void visualize_results(const viz_data &in_viz_data);

    /**
     *
     * Contains all of the boundary conditions of on the faces of this
     * Cell.
     *
     */
    std::vector<boundary_condition> BCs;

    /**
     *
     * The basis corresponding to this cell_manager.
     *
     */
    const BasisType *my_basis;

    /** @{
     *
     * All of the main local matrices of the element.
     *
     */
    Eigen::MatrixXd A, B, C, C2, D, E, E2, H, H0, H2, A0;
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
    Eigen::VectorXd exact_uhat, exact_u, exact_q, uhat_vec, u_vec, q_vec,
      q_star_dot_n_vec, grad_u_vec;
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
