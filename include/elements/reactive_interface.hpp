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
   * The boundary condition for semi-conductor and electrolyte interaction
   * problem. Since two different BCs can be applied to the same boundary,
   * we use bitwise numbering for different types of BCs.
   *
   */
  enum boundary_condition
  {
    not_set = 0,
    essential_rho_n = 1 << 0,
    essential_rho_p = 1 << 1,
    essential_rho_r = 1 << 2,
    essential_rho_o = 1 << 3,
    natural_rho_n = 1 << 4,
    natural_rho_p = 1 << 5,
    natural_rho_r = 1 << 6,
    natural_rho_o = 1 << 7,
    periodic = 1 << 8
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
     * @brief rhs_func of \f$\rho_n\f$ equation.
     */
    virtual double rho_n_rhs_func(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief rhs_func of \f$\rho_n\f$ equation.
     */
    virtual double rho_p_rhs_func(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief rhs_func of \f$\rho_r\f$ equation.
     */
    virtual double rho_r_rhs_func(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief rhs_func of \f$\rho_o\f$ equation.
     */
    virtual double rho_o_rhs_func(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief Dirichlet BC for \f$\rho_n\f$
     */
    virtual double gD_rho_n(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief Dirichlet BC for \f$\rho_p\f$
     */
    virtual double gD_rho_p(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief Dirichlet BC for \f$\rho_r\f$
     */
    virtual double gD_rho_r(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief Dirichlet BC for \f$\rho_o\f$
     */
    virtual double gD_rho_o(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief Dirichlet BC for \f$\rho_n\f$
     */
    virtual dealii::Tensor<1, dim>
    gN_rho_n(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief Dirichlet BC for \f$\rho_p\f$
     */
    virtual dealii::Tensor<1, dim>
    gN_rho_p(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief Dirichlet BC for \f$\rho_r\f$
     */
    virtual dealii::Tensor<1, dim>
    gN_rho_r(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief Dirichlet BC for \f$\rho_o\f$
     */
    virtual dealii::Tensor<1, dim>
    gN_rho_o(const dealii::Point<spacedim> &) = 0;

    /**
     *
     */
    virtual double lambda_inv2_S(
      const dealii::Point<spacedim> & = dealii::Point<spacedim>()) = 0;

    /**
     *
     */
    virtual double lambda_inv2_E(
      const dealii::Point<spacedim> & = dealii::Point<spacedim>()) = 0;

    /**
     * @brief Value of \f$\mu_n\f$.
     */
    virtual double
    mu_n(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) = 0;

    /**
     * @brief Value of \f$\mu_p\f$.
     */
    virtual double
    mu_p(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) = 0;

    /**
     * @brief Value of \f$\mu_r\f$.
     */
    virtual double
    mu_r(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) = 0;

    /**
     * @brief Value of \f$\mu_o\f$.
     */
    virtual double
    mu_o(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) = 0;

    /**
     * @brief Value of \f$\mu_n\f$.
     */
    virtual double
    alpha_n(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) = 0;

    /**
     * @brief Value of \f$\mu_p\f$.
     */
    virtual double
    alpha_p(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) = 0;

    /**
     * @brief Value of \f$\mu_r\f$.
     */
    virtual double
    alpha_r(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) = 0;

    /**
     * @brief Value of \f$\mu_o\f$.
     */
    virtual double
    alpha_o(const dealii::Point<spacedim> & = dealii::Point<spacedim>()) = 0;

    /**
     * @brief exact \f$\rho_n\f$
     */
    virtual double exact_rho_n(const dealii::Point<spacedim> &p) = 0;

    /**
     * @brief exact \f$\rho_p\f$
     */
    virtual double exact_rho_p(const dealii::Point<spacedim> &p) = 0;

    /**
     * @brief exact \f$\rho_r\f$
     */
    virtual double exact_rho_r(const dealii::Point<spacedim> &p) = 0;

    /**
     * @brief exact \f$\rho_o\f$
     */
    virtual double exact_rho_o(const dealii::Point<spacedim> &p) = 0;

    /**
     * @brief rhs_func of \f$\rho_n\f$ equation.
     */
    virtual dealii::Tensor<1, dim>
    exact_q_n(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief rhs_func of \f$\rho_n\f$ equation.
     */
    virtual dealii::Tensor<1, dim>
    exact_q_p(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief rhs_func of \f$\rho_n\f$ equation.
     */
    virtual dealii::Tensor<1, dim>
    exact_q_r(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief rhs_func of \f$\rho_n\f$ equation.
     */
    virtual dealii::Tensor<1, dim>
    exact_q_o(const dealii::Point<spacedim> &) = 0;

    /**
     * @brief Electric field.
     */
    virtual dealii::Tensor<1, dim>
    electric_field(const dealii::Point<spacedim> &p) = 0;

    /**
     * @brief the stabilization parameter.
     */
    virtual double tau(const dealii::Point<spacedim> &) = 0;
  };

  /**
   *
   *
   * This structure contains all of the data for visulaizing the solution.
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
             const std::string &in_filename,
             const std::vector<std::string> &in_var_names);

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
    const std::vector<std::string> my_var_names;
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
     * solve for 4 traces globally.
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
    unsigned n_trace_unkns_per_cell_dof() const;

    /**
     *
     * @brief Number of trace unknowns on the face.
     *
     */
    unsigned n_trace_unkns_per_face_dof() const;

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

    /**
     *
     * The dealii FE object which is used for visualization purposes.
     *
     */
    dealii::FESystem<dim> viz_fe;

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
   *   \right) &= f_n.
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
   *     + \nabla \cdot \mathbf q_n &= f_n.
   *   \end{aligned}
   * \f]
   * Let us use the notation: \f$c_n = \mu_n \alpha_n \lambda^{-2}\f$.
   * We satisfy the above equation in the weak sense, by testing it against
   * proper test functions:
   * \f[
   *   \begin{aligned}
   *     (\mu_n^{-1} \mathbf q_n, \mathbf p)
   *       + \langle \hat \rho_n, \mathbf p \cdot \mathbf n \rangle
   *       - (\rho_n, \nabla \cdot \mathbf p) &= 0, \\
   *     (\partial_t \rho_n, w)
   *       + \langle \boldsymbol H^*_n \cdot \mathbf n, w \rangle
   *       - (c_n \mathbf E \rho_n, \nabla w)
   *       - (\mathbf q_n , \nabla w) &= f_n(w), \\
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
   *     (c_n {\mathbf E}^* \cdot \mathbf n)
   *     \hat \rho_n + \mathbf q_n \cdot \mathbf n +
   *     \tau_{n} (\rho_n - \hat \rho_n ).
   * \end{aligned}
   * \f]
   * To solve this equation, we want to satisfy the following variational
   * form (we will add the time derivative term later):
   * \f[
   * \begin{aligned}
   *   \mu_n^{-1} (\mathbf q_n, \mathbf p)
   *     + \langle \hat \rho_n, \mathbf p \cdot \mathbf n \rangle
   *     - (\rho_n, \nabla \cdot \mathbf p) &= 0, \\
   *   \langle
   *     c_n({\mathbf E}^* \cdot \mathbf n)
   *     \hat \rho_n +
   *     \mathbf q_n \cdot \mathbf n + \tau_n (\rho_n - \hat \rho_n),w
   *   \rangle
   *   - (c_n \mathbf E \rho_n, \nabla w)
   *   - (\mathbf q_n , \nabla w) &= f_n(w).
   * \end{aligned}
   * \f]
   * And finally,
   * \f[
   * \begin{aligned}
   *   \mu_n^{-1} a_1(\mathbf q_n,\mathbf p) - b_1(\rho_n, \mathbf p)
   *     + c_1(\hat \rho_n, \mathbf p) &= 0, \\
   *   b_1^T(w, \mathbf q_n) + d_1(\rho_n, w)
   *                         - c_n d_2(\rho_n, w)
   *                         - e_1(\hat \rho_n, w)
   *                         + c_n e_2(\hat \rho_n, w)
   *                         &= F_n(w), \\
   * \end{aligned}
   * \f]
   * with:
   * \f[
   * \begin{gathered}
   *   a_1(\mathbf q_n , \mathbf p) = (\mathbf q_n, \mathbf p), \quad
   *   b_1(\rho_n , \mathbf p) = (\rho_n, \nabla \cdot \mathbf p), \quad
   *   c_1(\hat \rho_n , \mathbf p) = \langle\hat \rho_n, \mathbf p \cdot
   *                                  \mathbf n\rangle, \quad
   *   r_n(\mathbf p) = \langle g_{Dn} , \mathbf p \cdot \mathbf n \rangle,
   *   \\
   *   d_1(\rho_n,w) =
   *     \left\langle \tau_n \rho_n , w \right\rangle, \quad
   *   d_2(\rho_n,w) =
   *     (\mathbf E \rho_n , \nabla w) , \quad
   *   e_1(\hat \rho_n, w) =
   *     \left\langle \tau_n \hat \rho_n , w \right \rangle, \quad
   *   e_2(\hat \rho_n, w) =
   *     \left\langle
   *       \mathbf E^* \cdot \mathbf n \hat \rho_n , w
   *     \right \rangle.
   * \end{gathered}
   * \f]
   * Thus, the descritized forms of \f$\rho_n, \mathbf q_n\f$ can be obtained
   * using the following matrix relations:
   * \f[
   * \begin{gathered}
   * \mathbf q_{nh} = (\mu_n^{-1} A_1)^{-1}
   *                  \left(-C_1 \hat \rho_{nh} + B_1 \, \rho_{nh} \right), \\
   * \rho_{nh} = \left(B_1^T(\mu_n^{-1}A_1)^{-1} B_1 + D_1 - c_n D_2\right)^{-1}
   *             \left[ F_n +
   *             \left(B_1^T(\mu_n^{-1}A)^{-1} C_1 + E_1 - c_n E_2\right)
   *             \hat \rho_{nh} \right]
   * \end{gathered}
   * \f]
   * We also satisfy the conservation of the numerical flux:
   * \f[
   * \sum_{K\in \mathcal T_h}
   * \langle \boldsymbol H^*_h\cdot \mathbf n ,\mu \rangle_{\partial_K}
   * = \sum_{K\in \mathcal T_h}
   * \langle g_{Nn} ,\mu \rangle_{\partial_K}
   * \f]
   * Which results in:
   * \f[
   * \sum_{K \in \mathcal T_h}
   * c_1^T(q,\mu) + e_1^T(\rho_n, \mu) - h_1(\hat \rho_n, \mu)
   * + c_n h_2(\hat \rho_n, \mu) = l_n(\mu)
   * \f]
   * with
   * \f[
   * h_1(\hat \rho_n, \mu) = \langle \tau \hat \rho_n, \mu \rangle, \quad
   * h_2(\hat \rho_n, \mu) =
   * \langle \mathbf E^*\cdot \mathbf n \hat \rho_n, \mu \rangle.
   * \f]
   * Meanwhile, \f$l_n\f$ is the functional corresponding to the flux boundary
   * values that we apply on
   * \f$-\mu_n \alpha_n \rho_n \nabla \phi \cdot \mathbf n
   * - \mu_n \nabla \rho_n \cdot \mathbf n =
   * c_n (\mathbf E^* \cdot \mathbf n) \rho_n + \mathbf q \cdot \mathbf n\f$.
   * Similar equation that we mentioned here for electron density
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
   *     {\mathbf n}_{\Sigma,S} \cdot \mu_n\left(-\alpha_n \rho_n \nabla \Phi
   *       -\nabla \rho_n \right) = I_{et}(\rho_{n}-\rho_{n}^{e},\rho_{o}), \\
   *     {\mathbf n}_{\Sigma,S} \cdot \mu_p\left(-\alpha_p \rho_p \nabla \Phi
   *       -\nabla \rho_p \right) = I_{ht}(\rho_{p}-\rho_{p}^{e},\rho_{r}), \\
   *     {\mathbf n}_{\Sigma,E} \cdot \mu_r\left(-\alpha_r \rho_r \nabla \Phi
   *       -\nabla \rho_r \right) = I_{ht}(\rho_{p}-\rho_{p}^{e},\rho_{r})
   *                              - I_{et}(\rho_{n}-\rho_{n}^{e},\rho_{o}), \\
   *     {\mathbf n}_{\Sigma,E} \cdot \mu_o\left(-\alpha_o \rho_o \nabla \Phi
   *       -\nabla \rho_o \right) = - I_{ht}(\rho_{p}-\rho_{p}^{e},\rho_{r})
   *                                + I_{et}(\rho_{n}-\rho_{n}^{e},\rho_{o}).
   *   \end{gather}
   * \f]
   *
   * ### A note on implementation (For later developers)
   * In general, in this element, a face can have 4 degrees of freedom, i.e.
   * \f$\rho_n, \rho_p, \rho_r, \rho_o\f$. The order of degrees of freedom in
   * counting the unknowns is: \f$\rho_n, \rho_p, \rho_r, \rho_o\f$.
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
    static void visualize_results(const viz_data &in_viz_data);

    /**
     *
     * The basis corresponding to this cell_manager.
     *
     */
    const BasisType *my_basis;

    /**
     *
     * Contains all of the boundary conditions of on the faces of this
     * Cell.
     *
     */
    std::vector<boundary_condition> BCs;

    /**
     *
     * This bitset is designed for multiphysics problems. It is added when Ali
     * was solving the reactive interface problem. In this problem we have four
     * equations which were coupled at the reactive interface. So, there was no
     * need to perform the computations in the electrolyte for the density of
     * electrons or holes. On the other hand, it was not necessary to solve the
     * equations for density of oxidants or reductants in the semi-conductor.
     * Hence, we are adding this field which tells which equations should be
     * solved in which domain.
     *
     */
    boost::dynamic_bitset<> local_equation_is_active;

    /** @{
     *
     * All of the main local matrices of the element.
     *
     */
    Eigen::MatrixXd A1, B1, C1, D1, D2, E1, E2, H1, H2;
    ///@}

    /** @{
     *
     * @brief All of the element local vectors.
     *
     */
    Eigen::VectorXd Rn, Rp, Rr, Ro, Fn, Fp, Fr, Fo, Ln, Lp, Lr, Lo;
    ///@}

    /** @{
     *
     * @brief The exact solutions on the corresponding nodes.
     *
     */
    Eigen::VectorXd exact_rho_n, exact_rho_p, exact_rho_r, exact_rho_o;
    ///@}

    /** @{
     *
     * @brief The exact solutions on the corresponding nodes.
     *
     */
    Eigen::VectorXd exact_q_n, exact_q_p, exact_q_r, exact_q_o;
    ///@}

    /** @{
     *
     * @brief The exact solutions on the corresponding nodes.
     *
     */
    Eigen::VectorXd rho_n_vec, rho_p_vec, rho_r_vec, rho_o_vec;
    ///@}

    /** @{
     *
     * @brief The exact solutions on the corresponding nodes.
     *
     */
    Eigen::VectorXd q_n_vec, q_p_vec, q_r_vec, q_o_vec;
    ///@}

    /** @{
     *
     * @brief The source terms.
     *
     */
    Eigen::VectorXd gD_rho_n, gD_rho_p, gD_rho_r, gD_rho_o;

    /**
     *
     * @brief The boundary conditions.
     *
     */
    std::vector<dealii::Tensor<1, dim> > E_vec, E_star_vec;

    /**
     *
     * @brief The boundary conditions.
     *
     */
    Eigen::VectorXd f_n_vec, f_p_vec, f_r_vec, f_o_vec;

    /**
     *
     * @brief The std vector of gN's.
     *
     */
    std::vector<dealii::Tensor<1, dim> > gN_rho_n, gN_rho_p, gN_rho_r, gN_rho_o;
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
