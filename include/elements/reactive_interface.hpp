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

#ifndef REACTIVE_INTERFACE_HPP
#define REACTIVE_INTERFACE_HPP

namespace nargil
{
/**
 *
 *
 * The element to be used for solving the reactive_interface equation.
 * The first method of solving this is based on hybridized DG.
 *
 * \ingroup modelelements
 *
 *
 */
template <int dim, int spacedim = dim>
struct reactive_interface : public cell<dim, spacedim>
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
  reactive_interface(dealiiTriCell<dim, spacedim> *in_cell,
                     const unsigned id_num_,
                     base_basis<dim, spacedim> *base_basis);

  /**
   *
   * The destructor of the class.
   *
   */
  virtual ~reactive_interface() final;

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
    virtual double Ln_func(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief gD_func.
     */
    virtual double rho_n_BC(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief gD_func.
     */
    virtual double rho_p_BC(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief gD_func.
     */
    virtual double mu_n(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief gD_func.
     */
    virtual double mu_p(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief gD_func.
     */
    virtual double phi_s_BC(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief gN_func.
     */
    virtual dealii::Tensor<1, dim> gN_func(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief exact_u
     */
    virtual double exact_rho_n(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief exact_q
     */
    virtual dealii::Tensor<1, dim>
    exact_q_n(const dealii::Point<spacedim> &) = 0;

    /**
     *
     */
    virtual dealii::Tensor<2, dim>
    kappa_inv(const dealii::Point<spacedim> &) = 0;

    /**
     *
     */
    virtual double tau(const dealii::Point<spacedim> &) = 0;
  };

  /**
   *
   * This function is called from cell::create. This function cannot
   * be const, because the reactive_interface::my_manager is changed in this
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
  static void assign_data(reactive_interface *in_cell, data *in_data);

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
   * The data for this cell.
   *
   */
  data *my_data;

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
    typedef typename reactive_interface::template hdg_manager<hdg_polybasis>
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
     * equations to obtain them. For example in reactive_interface equation we
     * solve
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
   * Here, we simulate a reactive interface \f$(\Sigma)\f$ between a
   * semiconductor (\f$\Omega_S\f$) and an elecrolyte medium (\f$\Omega_E\f$).
   * We want to solve the coupled equations in the domain \f$\Omega = \Omega_S
   * \cup \Omega_E\f$. The densities of elctrons (\f$\rho_n\f$) and holes
   * (\f$\rho_p\f$) in the semiconductor are coupled with the densities of
   * reductants (\f$\rho_r\f$) and oxidants (\f$\rho_o\f$) in the electrolyte.
   * Hence, a coupled system of advection-diffusion equations governs these
   * densities. The final goal here is to solve these coupled equations
   * implicitly. Meanwhile, the advection veclocities in these equations are
   * defined based on the electrical force (\f$\mathbf E = -\nabla \phi\f$),
   * with \f$\phi\f$ being the electric potential. Poisson's equation governs
   * the electric potential as follows:
   * \f[
   *   \begin{aligned}
   *   \lambda^{-2} \mathbf E + \nabla \phi &= 0, \\
   *   \nabla \cdot \mathbf E &= f,
   *   \end{aligned}\tag{1}
   * \f]
   * The solution technique to this equation is discussed in the
   * diffusion::hdg_manager (with \f$\kappa = \lambda^2\f$). On the other
   * hand, the advection-diffusion equations governing the densities can
   * be written as:
   * \f[
   *   \begin{aligned}
   *   \partial_t \rho_n -
   *   \nabla \cdot \mu_n
   *   \left(
   *     \alpha_n \rho_n \nabla \phi + \nabla \rho_n
   *   \right) &= L_n.
   *   \end{aligned}
   * \f]
   * To know about each of the above unknowns and parameters, one can visit
   * <a href="http://dx.doi.org/10.1016/j.jcp.2016.08.026"> this article</a>.
   * We solve this equation by writing it in terms of a first order system:
   * \f[
   *   \begin{aligned}
   *     \mu_n^{-1}\mathbf q_n
   *       + \nabla \rho_n &= 0, \\
   *     \partial_t \rho_n
   *     + \nabla \cdot (\mu_n \alpha_n \lambda^{-2} \mathbf E \rho_n)
   *     + \nabla \cdot \mathbf q_n &= L_n.
   *   \end{aligned}
   * \f]
   * We satisfy this equation in the weak sense, by testing it against proper test
   * functions:
   * \f[
   *   \begin{aligned}
   *     (\mu_n^{-1} \mathbf q_n, \mathbf p)
   *       + \langle \hat \rho_n, \mathbf p \cdot \mathbf n \rangle
   *       - (\rho_n, \nabla \cdot \mathbf p) &= 0, \\
   *     (\partial_t \rho_n, w)
   *       + \langle \boldsymbol H^*_n \cdot \mathbf n, w \rangle
   *       - (\mu_n \alpha_n \lambda^{-2} \mathbf E^* \rho_n, \nabla w)
   *       - (\mathbf q_n , \nabla w) &= L_n(w), \\
   *   \end{aligned} \tag{2}
   * \f]
   * Here, \f$\mathbf E^*\f$ denotes the numerical flux corresponding to
   * \f$\mathbf E\f$. This is a known quantity, since we have solved Eq. (1)
   * proior to Eq. (2) and we have \f$\mathbf E^*\f$.
   * Meanwhile, we use the follwoing definitions
   * for the numerical flux \f$\boldsymbol H^*_n\f$:
   * \f[
   * \begin{aligned}
   *   {\boldsymbol H}^*_n \cdot \mathbf n &=
   *     (\mu_n \alpha_n \lambda^{-2} {\mathbf E}^* \cdot \mathbf n)
   *     \hat \rho_n + \mathbf q_n \cdot \mathbf n +
   *     \tau_{n} (\rho_n - \hat \rho_n ).
   * \end{aligned}
   * \f]
   * To solve this equation, we want to satisfy the following variational
   * form (we will add the time derivative term later):
   * \f[
   * \begin{aligned}
   *   (\mu_n^{-1} \mathbf q_n, \mathbf p)
   *     + \langle \hat \rho_n, \mathbf p \cdot \mathbf n \rangle
   *     - (\rho_n, \nabla \cdot \mathbf p) &= 0, \\
   *   \langle
   *     (\mu_n \alpha_n \lambda^{-2} {\mathbf E}^* \cdot \mathbf n)
   *     \hat \rho_n +
   *     \mathbf q_n \cdot \mathbf n + \tau_n (\rho_n - \hat \rho_n),w
   *   \rangle
   *   - (\mu_n \alpha_n \lambda^{-2} \mathbf E^* \rho_n, \nabla w)
   *   - (\mathbf q_n , \nabla w) &= L_n(w).
   * \end{aligned}
   * \f]
   * And finally,
   * \f[
   * \begin{aligned}
   *   a_1(\mathbf q_n,\mathbf p) - b_1(\rho_n, \mathbf p)
   *     + c_1(\hat \rho_n, \mathbf p) &= 0, \\
   *   b_1^T(w, \mathbf q_n) + d_1(\rho_n, w) + e_1(\hat \rho_n, w) &= L_n(w),
   * \\
   * \end{aligned}
   * \f]
   * with:
   * \f[
   * \begin{gathered}
   *   a_1(\mathbf q_n , \mathbf p) = (\mathbf q_n, \mathbf p), \quad
   *   b_1(\rho_n , \mathbf p) = (\rho_n, \nabla \cdot \mathbf p), \quad
   *   c_1(\hat \rho_n , \mathbf p) = \langle\hat \rho_n, \mathbf p \cdot
   *                                  \mathbf n\rangle,
   *   \\
   *   d_1(\rho,w) =
   *     \left\langle \tau_n \rho_n , w \right\rangle
   *     - (\mu_n \alpha_n \lambda^{-2} \mathbf E^* \rho_n , \nabla w) , \quad
   *   e_1(\hat \rho, w) =
   *     \left\langle
   *       (\mu_n \alpha_n \lambda^{-2} \mathbf E^* \cdot \mathbf n
   *        - \tau_n) \hat \rho_n , w
   *     \right \rangle
   * \end{gathered}
   * \f]
   * Similar equations that we mentioned here for electron density
   * \f$(\rho_n)\f$ holds for holes \f$(\rho_p)\f$:
   * \f[
   *   \partial_t \rho_p - \nabla \cdot \mu_p
   *   \left(
   *     \alpha_p \rho_p \nabla \phi + \nabla \rho_p
   *   \right) = L_p.
   * \f]
   * According to the article we mentioned above, \f$\mu_n = 1350
   * \text{ V}/\text{cm}^2\f$, \f$\mu_p = 480 \text{ V}/\text{cm}^2\f$,
   * \f$\alpha_n = -1\f$, and \f$\alpha_p = 1\f$.
   *
   * On the other hand in the electrolyte region, similar equations hold
   * for the densities of reductants \f$(\rho_r)\f$ and oxidants
   * \f$(\rho_o)\f$, i.e.:
   * \f[
   *   \begin{gathered}
   *   \partial_t \rho_r - \nabla \cdot \mu_r
   *   \left(
   *     \alpha_r \rho_r \nabla \phi + \nabla \rho_r
   *   \right) = L_r, \\
   *   \partial_t \rho_o - \nabla \cdot \mu_o
   *   \left(
   *     \alpha_o \rho_o \nabla \phi + \nabla \rho_o
   *   \right) = L_o.
   *   \end{gathered}
   * \f]
   * Despite the quite simple structure of the governing equations that
   * we mentioned above, the boundary conditions of these equations are
   * a set of nonlinear interface conditions:
   * \f[
   *   \begin{gather}
   *   \end{gather}
   * \f]
   *
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
    hdg_manager(const reactive_interface<dim, spacedim> *,
                const BasisType *in_basis);

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
     * Called from compute_errors().
     *
     */
    void compute_my_errors(std::vector<double> *sum_of_L2_errors);

    /**
     *
     * This function is used to interpolate the function f to the trace
     * degrees of freedom of the element.
     *
     */
    static void assign_BCs(reactive_interface *in_cell, BC_Func f);

    /**
     *
     * This function is used to interpolate the function f to the trace
     * degrees of freedom of the element.
     *
     */
    static void interpolate_to_trace(reactive_interface *in_cell);

    /**
     *
     * Called from outside of the function to interpolate the input function
     * to the interior of the given cell.
     *
     */
    static void interpolate_to_interior(reactive_interface *in_cell);

    /**
     *
     * Fills the visualization vector of the element.
     *
     */
    static void fill_viz_vector(reactive_interface *in_cell,
                                distributed_vector<dim, spacedim> *out_vec);

    /**
     *
     * Fills the refinement vector of the element.
     *
     */
    static void fill_refn_vector(reactive_interface *in_cell,
                                 distributed_vector<dim, spacedim> *out_vec);

    /**
     *
     * This function, sets source term, Dirichlet and Neumann BC functions.
     *
     */
    static void set_source_and_BCs(reactive_interface *in_cell);

    /**
     *
     * compute_my_local_unkns
     *
     */
    static void compute_local_unkns(reactive_interface *in_cell,
                                    const double *trace_sol);

    /**
     *
     * Called from outside of the class to assemble global matrices and
     * rhs vector corresponding to this element.
     *
     */
    static void
    assemble_globals(reactive_interface *in_cell,
                     solvers::base_implicit_solver<dim, spacedim> *in_solver);

    /**
     *
     * Compute the error of u and q, based on the exact_u and exact_q.
     * As a result the function static set_exact_local_dofs should be called
     * before calling this function.
     *
     */
    static void compute_errors(reactive_interface *in_cell,
                               std::vector<double> *sum_of_L2_errors);

    /**
     *
     * This function writes the vtk output.
     *
     */
    static void
    visualize_results(const dealii::DoFHandler<dim, spacedim> &dof_handler,
                      const LA::MPI::Vector &visual_solu,
                      unsigned const &time_level);

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

#include "../../source/elements/reactive_interface.cpp"

#endif
