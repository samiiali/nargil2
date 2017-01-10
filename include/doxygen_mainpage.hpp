namespace nargil
{
/**
 * @mainpage An Intro to nargil:
 *
 * <B>nargil</B> is a set of reuasable software tools for finite element
 * analysis of different problems.
 * It is not a library nor a program. It is developed
 * as a solver for the different physical problems that came up during its
 * development cycle! However, the main advantage of this toolset is its
 * extremely modular format. It is written in C++ and uses template
 * meta-programming to resolve all types in the compile time. As such, when
 * the program is compiled to for example to solve the advection-diffusion
 * equation with hybridized discontinuous Galerkin method,
 * the produced bindary code cannot solve the same equation with some
 * H-div conforming element like Raviart-Thomas. Althouhg there is a
 * possibility to compile the software to use RT elements for the
 * advection-diffusion equation. So, I cannot call it a
 * finite element program, because a program is based on
 * the user interaction at run-time. Albeit, we continue to call the developed
 * software a program.
 *
 * nargil depends extensively on deal.II and PETSc libraries.
 *
 * ### The nargil's Workflow:
 *
 * Here we review the workflow of the program.
 *
 * 1. <B>Creating a nargil::mesh object</B>: In the first phase we create a
 *    nargil::mesh object. To this end we need to call the mesh::mesh()
 *    with three arguments, i.e. the MPI_Comm, number of threads and a flag
 *    which states if the mesh is going to be <I>h</I>-adaptive or not.
 *    The constructor of nargil::mesh, also creates an empty deal.II
 *    triangulation, which we later assign a mesh to it in
 *    mesh::generate_mesh(). For this purpose,
 *    we write a function which generates a deal.II triangulation and pass it
 *    to the mesh object through mesh::generate_mesh(). generate_mesh()
 *    will also call mesh::init_cell_ID_to_num(). This function fills
 *    two std::maps in the mesh object. The first std::map
 *    (mesh::owned_cell_ID_to_num) maps the deal.II
 *    cell unique ID of all owned cells to a unique integral number. The second
 *    std::map (mesh::ghost_cell_ID_to_num), does the same thing for all
 *    ghost cells. Therefore, the mesh object contains the deal.II
 *    triangulation, MPI_Comm, and two std::maps which are mapping unique
 *    cell IDs to integer numbers.
 *
 * 2. <B>Create a nargil::model object </B>: The constructor of the model
 *    object takes mesh object and creates a model object. As
 *    its template parameter we also support a derived typename of cell
 *    class to the model object. This template parameter will be used
 *    as the type of the physics that we consider in our model (such as
 *    diffusion). In the model object we store a
 *    pointer to a const nargil::mesh (this is done in the
 *    constructor of the model).
 *
 * 3. <B>Create a derived class of nargil::base_basis and pass it to model</B>:
 *    Now that we have the model object, we derive a class from
 *    base_basis, which serves as the basis for the cells in the mesh.
 *    We use the created
 *    basis as the argument to the function model::init_model_elements.
 *    In this
 *    function, we fill the two std::vectors of std::unique_ptrs to
 *    cell's. These two std::vectors are model::all_owned_cells
 *    and model::all_ghost_cells. To create cells based on the supported
 *    basis, we call cell::create. This function takes an iterator to the
 *    deal.II cell, cell number, the basis, and the model. Then, it calls
 *    the cell constructor corresponding to the template parameter of the
 *    nargil::model, i.e. ModelEq.
 *    A cell based on ModelEq will be ceated, and
 *    the basis will be assigned to it. Next, cell::create calls the
 *    assign_manager function corresponding to the ModelEq class.
 *    For example for diffusion class, diffusion::assign_manager
 *    will be called. This function assigns the type of the
 *    nargil::cell_manager to the cell. The cell_manager is a class that
 *    performs all of the main tasks of the elements. For example
 *    assembling the element matrices or doing the postprocessing
 *    tasks. The type of the cell_manager is decided based on the type
 *    of basis, which is supported to cell::assign_manager. So, there
 *    should be a well-defined map from basis types to cell_manager types. By
 *    this, we mean the same basis cannot map to two different cell_manager's.
 *
 * 4. <B>Assigning the boundary conditions</B>: After having the model and
 *    all the elements, we create a function which assigns the boundary
 *    conditions to the elements. For example when we have a diffusion
 *    equation which we want to solve it with hybridized DG, we take a
 *    diffusion::hdg_manager as its input. The diffusion::hdg_manager inherits
 *    two std::vectors from hybridized_cell_manager. One is
 *    hybridized_cell_manager::BCs
 *    and the other is hybridized_cell_manager::dofs_ID_in_all_ranks. The former
 *    contains "number of faces" enums of type boundary_condition, and
 *    the latter contains "number of faces" dynamic_bitsets; the size of each of
 *    these dynamic_bitsets are equal to the number of dofs. For example in
 *    the diffusion equation \f$\nabla \cdot (\kappa \nabla u) = f\f$, and the
 *    hybridized DG solver, we only have \f$\hat u\f$ as the face dof, and
 *    each dynamic_bitset has only one slot. Then, a closed dof is identified
 *    with the corresponding bit equal to 0, and the open dof is identified with
 *    the bit equal to 1.
 *
 * 5. <B>Counting the global unknowns</B>: We do so by calling the
 *    model::count_globals(). The argument that we pass to count globals is
 *    a derived type of dof_counter. Each derived type of dof_counter is
 *    compatible with one predefined derived type of cell_manager. For example
 *    implicit_hybridized_numbering is compatible with hybridized_cell_manager.
 *    In hybridized_cell_manager we store all the data required for numbering of
 *    each global dof of a cell with the global unknowns on its faces (such as
 *    HDG).
 */
}
