/*!
 * \mainpage
 * \section intro_sec 1. Introduction
 * This program is part of our effort in the Computational Hydraulics Group,
 * to solve Green-Naghdi equations using hybrid discontinuous Galerkin
 * method. Green-Naghdi equation can be used to describe the propagation of
 * fully nonlinear weakly dispersive water waves. However, since these equations
 * are complicated, we start our effort by solving a linear equation which
 * can be used for solving the problems with a flat bottom. In the following
 * figure, three different length scales (water height scale: \f$H_0\f$,
 * wave amlitude scale: \f$a_{\text{surf}}\f$, bathymetry scale
 * \f$a_{\text{bott}}\f$) and three dimensionless parameters are
 * shown:
 * <img src="Images/Waves.png" width=500em />
 * <h3 style="text-align:center; font-size:1em;">
 * Different length scales and dimensionless parameters, which characterize
 * the regime of the flow.
 * </h3>
 * In the first step we assume \f$\varepsilon = \beta = 0\f$ and
 * \f$\mu \ll 1 \f$. The nondimnesionalized version of this equation,
 * can be obtained by simplifying the original Green-Naghdi equations.
 *
 * \section equation_sec 2. Dispersive Linear SWE for Flat Bottom
 * We want to solve a system of two equations. The first equation is
 * always obtained by setting the normal velocity of water particles at the
 * water surface equal to zero.
 * \f[
 * \partial_t \zeta + \nabla \cdot (h \mathbf V) = 0.
 * \f]
 * In the above nondimensional equation, \f$h = 1+\varepsilon \zeta + \beta
 * b\f$. In our linearized flat bottom approximation, \f$\varepsilon = \beta =
 * 0\f$. Hence, the equation would reduce to:
 * \f[
 * \label{eq1}\partial_t \zeta + \nabla \cdot \mathbf V = 0.\tag{1}
 * \f]
 * The corresponding equation with dimension is:
 * \f[
 * t_0 \partial_t \zeta + \frac L {V_0} \nabla \cdot \mathbf V = 0.
 * \f]
 * Where, \f$t_0 = {L}/{\sqrt{g h_0}}\f$, \f$V_0 =
 * a_{\text{surf}} \sqrt{g/ {h_0}}\f$. However, we are not going to use
 * the above equation with dimension; because \f$\zeta \ll h\f$, and the
 * dimensionless equation is a better option to solve, I think!
 *
 * The second equation is the momentum conservation equations, which can
 * be found in textbooks. So, the equations that we solve here are
 * \f[
 * \left\{
 * \begin{aligned}
 * & \partial_t \zeta+\nabla \cdot \mathbf V = 0 \\
 * & \partial_t \mathbf V - \frac{\mu}{3} \nabla
 *       (\nabla \cdot (\partial_t \mathbf V)) + \nabla \zeta = 0
 * \end{aligned}
 * \right.
 * \tag{2}
 * \f]
 * Note that, if we drop the term with \f$\mu\f$ coefficient, we get the well
 * known acoustic wave equation, which reads as: \f$ \partial^2_t \zeta -
 * \nabla^2 \zeta = 0 \f$.
 *
 * \subsection subsec_02_01 2.1. Solving Eqs. (2) without Operator Splitting
 * We intorduce a new unknown \f$(q)\f$ in the above equations, which can be
 * obtained through: \f$q + \nabla \cdot (\partial_t \mathbf V) = 0\f$. Now,
 * we want to solve the following system of equations simultaneously:
 * \f[
 * \left\{
 * \begin{aligned}
 * & q + \nabla \cdot (\partial_t \mathbf V) = 0 \\
 * & \partial_t \zeta+\nabla \cdot \mathbf V = 0 \\
 * & \partial_t \mathbf V - \nabla \left( \frac{\mu}{3} q + \zeta \right) = 0
 * \end{aligned}
 * \right.
 * \f]
 * The last two equations in the above relation can be written in conservation
 * form:
 * \f[
 * \partial_t \mathbf u + \nabla \cdot \mathbb F(\mathbf u) = 0,
 * \quad \text{with }
 * \mathbf u = \begin{pmatrix} \zeta \\ V_1 \\ V_2 \end{pmatrix},
 * \text{ and }
 * \mathbb F(\mathbf u) =
 * \begin{pmatrix}
 * V_1 & V_2 \\ c_0 q + \zeta & 0 \\ 0 & c_0 q + \zeta
 * \end{pmatrix} =
 * \begin{pmatrix}
 * \mathbf V \\ (c_0 q + \zeta) \mathbf I
 * \end{pmatrix}
 * \f]
 * where, we have shown numerical fluxes with \f$\widehat {(\cdot)}\f$ and
 * \f$ c_0 = \mu/3 \f$. The corresponding variational form may be written
 * as:
 * \f[
 * \left\{
 * \begin{aligned}
 * & \langle \partial_t \widehat {\mathbf V} \cdot \mathbf n , p \rangle -
 *   (\partial_t \mathbf V , \nabla p) + (q , p) = 0 \\
 * & (\partial_t \zeta , \xi) +
 *   \langle \widehat {\mathbf V} \cdot \mathbf n , \xi \rangle
 *   - (\mathbf V , \nabla \xi)  = 0 \\
 * & (\partial_t \mathbf V, \mathbf U) -
 *   \langle \widehat {c_0 q} +
 *   \widehat {\,\zeta\,} , \mathbf U \cdot \mathbf n \rangle +
 *   (c_0 q + \zeta , \nabla \cdot \mathbf U) = 0
 * \end{aligned}
 * \right.
 * \f]
 * Now, for the conservative part, we use the numerical flux as below:
 * \f[
 * \widehat {\mathbb F}(\mathbf u) \cdot \mathbf n =
 *      \mathbb F(\hat {\mathbf u}) \cdot \mathbf n +
 *      \boldsymbol \tau(\mathbf u - \hat {\mathbf u})
 * \f]
 * With \f$\boldsymbol \tau\f$ defined based on the eigenvectors and
 * eigenvalues of the matrix: \f$\mathbf A = \partial \mathbb
 * F(\hat {\mathbf u}) / \partial \hat {\mathbf u} \cdot \mathbf n \f$.
 * We can also define the flux \f$\widehat {c_0 q}\f$ via the following
 * definition of the flux:
 * \f[
 * \widehat{c_0 q} = c_0 q +
 *   \sigma_1 (\mathbf V - \hat {\mathbf V}) \cdot \mathbf n
 * \f]
 *
 * \subsection subsec_02_02 2.2. Solving Eq. (2) with Operator Splitting
 * Obtaining a stable method from the technique described in the
 * previous section is not straightforward. As a result, we are
 * using Strang operator splitting technique to split the system (2)
 * to the following systems:
 * \f[
 * \text{Eq a: }
 * \left\{
 * \begin{aligned}
 * & \partial_t \zeta+\nabla \cdot \mathbf V = 0 \\
 * & \partial_t \mathbf V = 0
 * \end{aligned}
 * \right.
 * ,\quad
 * \text{Eq b: }
 * \left\{
 * \begin{aligned}
 * & \partial_t \zeta = 0 \\
 * & \partial_t \mathbf V + \left(\mathcal I + \mu \mathcal T\right)^{-1}
 *                          \nabla \zeta = 0
 * \end{aligned}
 * \right.
 * \tag{3}
 * \f]
 * Here, \f$ \mathcal I\f$ is identity operator and \f$\mathcal T\f$ is
 * a pseudo-differential operator, defined as below:
 * \f[
 * \mathcal T \mathbf q = -\frac 1 3 \nabla(\nabla \cdot \mathbf q).
 * \f]
 *
 * \subsubsection subsub_eq3a 2.2.1. Applying HDG to Eq. (3a)
 * Now, let us go back and look at Eqs. (3a). In this equation,
 * \f$\mathbf V\f$ is not varying with time, and for solving
 * the first equation in this system, we can use an explicit
 * or implicit time integration. The variational form reads as:
 * \f[
 * \left\{
 * \begin{aligned}
 * & (\partial_t \zeta, \xi) +
 *   \langle \widehat{\mathbf V} \cdot \mathbf n , \xi \rangle -
 *   (\mathbf V , \nabla \xi) = 0 \\
 * & \partial_t \mathbf V = 0
 * \end{aligned}
 * \right.
 * \f]
 * As usual, we write this equation in the conservation form:
 * \f[
 * \partial_t \mathbf v + \nabla \cdot \mathbb F(\mathbf v) = 0,
 * \tag{4}
 * \f]
 * \f[
 * \quad \text{with }
 * \mathbf v = \begin{pmatrix} \zeta \\ V_1 \\ V_2 \end{pmatrix},
 * \text{ and }
 * \mathbb F(\mathbf v) =
 * \begin{pmatrix}
 * V_1 & V_2 \\ 0 & 0 \\ 0 & 0
 * \end{pmatrix} =
 * \begin{pmatrix}
 * \mathbf V \\ 0 \mathbf I
 * \end{pmatrix}
 * \f]
 * For \f$\widehat{\mathbf V}\f$, we use
 * \f$\widehat{\mathbf V} \cdot \mathbf n =
 *     \hat {\mathbf V} \cdot \mathbf n +
 *     \boldsymbol \tau (\mathbf V - \hat {\mathbf V}). \f$
 *
 * Despite this lengthy explanation, equation (3a) can be solved simply by
 * integrating in time. Since, we have \f$\mathbf V\f$ as the initial condition
 * and it does not change with time, we only need to use an implicit or
 * explicit technique to obtain \f$\zeta\f$ in the next time step. Refer
 * to GN_eps_0_beta_0_stage1::calculate_matrices() to know more about the
 * formulation.
 *
 * \subsubsection subsub_eq3b 2.2.2. Applying HDG to Eq. (3b)
 * We first define the auxiliary variable \f$\mathbf q_1\f$, which is connected
 * to \f$\nabla \zeta\f$ through:
 * \f$(\mathcal I + \mu \mathcal T) \mathbf q_1 = \nabla \zeta\f$. Then, our
 * main goal is to solve for \f$\mathbf q_1\f$, having \f$\nabla \zeta\f$.
 * So, for example in an explicit method, we solve for \f$\mathbf q_1\f$ having
 * \f$\nabla \zeta\f$ from the previous time step.
 * Then, we put this \f$\mathbf q_1\f$ which
 * correspond to the previous time step in Eq. (3b) and compute
 * \f$\mathbf V\f$ of the current time step.
 *
 * Now, let us take a look at operator \f$\mathcal T\f$.
 * This operator contains the gradient of divergence of a vector. A similar
 * operator also comes up in
 * <a href="https://en.wikipedia.org/wiki/Linear_elasticity"> Navier-Cauchy
 * equation</a>. In indicial form, we can write: \f$u_{i,ij} =
 * (u_{i,i}\delta_{jk})_{,k}\f$. In other words: \f$\nabla (\nabla
 * \cdot \mathbf q) = \nabla \cdot \left(
 * (\nabla \cdot \mathbf q) \mathbf I \right)\f$. So, we define:
 * \f[
 * \begin{aligned}
 *   \nabla \zeta &= \mathbf q_1 - \frac \mu 3 \nabla \cdot
 * (q_2\mathbf I) \\
 *   {q_2} &= \nabla \cdot \mathbf q_1
 * \end{aligned}
 * \f]
 * which is to be solved along with Eq. (3b). So, the whole system
 * of equations corresponding to Eq. (3b) can be written as:
 * \f[
 * \left\{
 * \begin{aligned}
 * &
 *   \left. \begin{aligned}
 *      &\partial_t \zeta = 0 \\
 *      &\partial_t \mathbf V + \mathbf q_1= 0
 *    \end{aligned}
 *   \right\} \text {Original PDE} \\
 * &
 *   \left. \begin{aligned}
 *      & \mathbf q_1 - \frac \mu 3 \nabla \cdot
 *        (q_2\mathbf I) = \nabla \zeta \\
 *      & q_2 - \nabla \cdot \mathbf q_1 = 0
 *    \end{aligned}
 *   \right\} \text{Auxiliary equations}
 * \end{aligned}
 * \right.
 * \tag{3b (expanded)}
 * \f]
 * Now, the plan is computing \f$\mathbf q_1\f$ from the auxiliary equations
 * and substituting it in the original PDE and solve for \f$\mathbf V\f$. The
 * variational form corresponding to the auxiliary equations can be written as:
 * \f[
 *   \left\{
 *     \begin{aligned}
 *       & (\mathbf q_1, \mathbf U)
 *         - \frac \mu 3 \langle \widehat {\mathsf {q_2}} \cdot
 *                             \mathbf n,\mathbf U
 *                     \rangle
 *         + \frac \mu 3 (q_2 \mathbf I , \nabla \mathbf U)
 *         = (\nabla \zeta , \mathbf U)
 *     \\
 *       & (q_2 , p_2) -
 *         \langle \hat{\mathbf q}_1 \cdot \mathbf n, p_2 \rangle +
 *         (\mathbf q_1 , \nabla p_2) = 0
 *     \end{aligned}
 *   \right.
 * \f]
 *
 * I assume, in these equations
 * we should define the flux \f$\widehat{\mathsf{q_2}} \cdot \mathbf n\f$ in
 * terms of the trace of \f$\hat {\mathbf q}_1\f$. That is why, I suggest
 * using the definition of the flux as:
 * \f[
 *    \widehat{\mathsf{q_2}} \cdot \mathbf n =
 *      (q_2 \mathbf I) \cdot \mathbf n +
 *    \boldsymbol \tau (\mathbf q_1 - \hat {\mathbf q}_1)
 * \f]
 * For now, let us assume \f$\boldsymbol \tau = \mathbf I\f$.
 *
 * <b> Note: </b> One of the main concerns that I have now
 * is applying the boundary conditions on \f$\mathbf q_1\f$ and \f$q_2\f$.
 * I understand that we should apply the boundary conditions weakly, but it
 * depends on what type of boundary data we have! So, for now, let us continue
 * with periodic boundary condition. This boudnary condition will be applied
 * by setting: \f$\langle[\![ \widehat{\mathsf q_2}
 *                \cdot \mathbf n ]\!] ,\mu \rangle = 0\f$, everywhere. Also,
 * the values of \f$\hat {\mathbf q}_1\f$ on the corresponding boundaries
 * will be set equal to each other.
 *
 * Follow the rest of formulation in the description of the
 * GN_eps_0_beta_0_stage2::calculate_matrices. To see some numerical examples
 * check the \ref GN_0_0_stage2_page "numerical examples page".
 *
 */
