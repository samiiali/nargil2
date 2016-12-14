/*! \page diffusion_tests Extremely Anisotropic and Nonhomogeneous Diffusion
 * Here, we present a set of numerical examples, which are solved
 * for an extremely anisotropic diffusion problem. This problem is
 * solved in the context of plasma physics.
 * Consider the well known diffusion equation:
 * \f[
 *   -\nabla \cdot (K \cdot \nabla T) = Q,
 *   \quad \text{in } \Omega = (-0.5,0.5)^2.
 * \f]
 * Here, \f$K\f$ is the diffusivity tensor, and in order to define
 * it, we consider the magnetic field \f$\mathbf B\f$, as below:
 * \f[
 *   \mathbf B = \{\pi \cos (\pi x)\sin (\pi y)
 *                 ,-\pi\sin(\pi x) \cos (\pi y)\}
 * \f]
 * We then define \f$\mathbf b = \mathbf B / |\mathbf B|\f$.
 * Now \f$K\f$ is defined as:
 * \f[
 *   K = I + (\epsilon^{-1} -1) \mathbf b \otimes \mathbf b
 * \f]
 * As a result, the source term \f$Q\f$ can be obtained as:
 * \f[
 *   Q = 2 \pi^2 \cos(\pi x) \cos(\pi y).
 * \f]
 */
