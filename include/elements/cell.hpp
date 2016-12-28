#include <memory>
#include <type_traits>

#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria_accessor.h>

#ifndef CELL_CLASS_HPP
#define CELL_CLASS_HPP

/**
 * \defgroup modelelements Model Elements
 * \brief This group contains different model elements and the relevant
 * structures used to solve different model problems.
 */

namespace nargil
{

//
//
// Forward decleration of BaseModel.
struct BaseModel;

//
//
/**
 * \brief The enum which contains different boundary conditions.
 */
enum class BC
{
  not_set = ~(1 << 0),
  essential = 1 << 0,
  flux_bc = 1 << 1,
  periodic = 1 << 2,
  in_out_BC = 1 << 3,
  inflow_BC = 1 << 4,
  outflow_BC = 1 << 5,
  solid_wall = 1 << 6,
};

//
//
/**
 * \brief Contains most of the required data about a generic
 * element in the mesh.
 * \ingroup modelelements
 */
template <int dim, int spacedim = dim> struct Cell
{
  //
  //
  /**
   * @brief The deal.II cell iterator type.
   */
  typedef dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim> >
    dealii_cell_type;

  //
  //
  /**
   * @name deal.II FEValues used in the class
   */
  ///@{
  typedef std::unique_ptr<dealii::FEValues<dim> > FE_val_ptr;
  typedef std::unique_ptr<dealii::FEFaceValues<dim> > FEFace_val_ptr;
  ///@}

  //
  //
  /**
   * \details
   * We remove the default constructor to avoid uninitialized creation of Cell
   * objects.
   */
  Cell() = delete;

  //
  //
  /**
   * \details
   * The constructor of this class takes a deal.II cell and creates the cell.
   */
  Cell(dealii_cell_type &inp_cell, const unsigned id_num_, BaseModel *model_);

  //
  //
  /**
   * \details
   * We remove the copy constructor of this class to avoid unnecessary copies
   * (specially unintentional ones).
   */
  Cell(const Cell &inp_cell) = delete;

  //
  //
  /**
   * \details
   * We need a move constructor, to be able to pass this class as function
   * arguments efficiently. Maybe, you say that this does not help efficiency
   * that much, but we are using it for semantic constraints.
   * \param inp_cell An object of the \c Cell_Class type which we steal its
   * guts.
   */
  Cell(Cell &&inp_cell) noexcept;

  //
  //
  /**
   * @brief The destructor of the class.
   */
  virtual ~Cell();

  //
  //
  /**
   * @brief This is the factory function which creates a cell of type
   * ModelEq (the template parameter). This function is called by
   * Model::init_mesh_containers.
   */
  template <typename ModelEq>
  static std::unique_ptr<Cell<dim, spacedim> >
  create(dealii_cell_type &inp_cell, const unsigned id_num_, BaseModel *model_);

  //
  //
  /**
   * This function Moves the dealii::FEValues and dealii::FEFaceValues
   * objects between different elements and faces. dealii::FEValues are not
   * copyable objects. They also
   * do not have empty constructor. BUT, they have (possibly
   * very efficient) move assignment operators. So, they should
   * be created once and updated afterwards. This avoids us from
   * using shared memory parallelism, because we want to create
   * more than one instance of this type of object and update it
   * according to the element in action. That is why, at the
   * beginning of assemble_globals and internal unknowns
   * calculation, we create std::unique_ptr to our FEValues
   * objects and use the std::move to move them along to
   * next elements.
   */
  void attach_FEValues(FE_val_ptr &,
                       FEFace_val_ptr &,
                       FE_val_ptr &,
                       FEFace_val_ptr &);

  //
  //
  /**
   * The opposite action of Cell::attach_FEValues.
   */
  void detach_FEValues(FE_val_ptr &,
                       FEFace_val_ptr &,
                       FE_val_ptr &,
                       FEFace_val_ptr &);

  //
  //
  /**
   * \details Updates the FEValues which are connected to the current element
   * (not the FEFaceValues.)
   */
  void reinit_cell_fe_vals();

  //
  //
  /**
   * \details Updates the FEFaceValues which are connected to a given face of
   * the current element.
   */
  void reinit_face_fe_vals(unsigned);

  //
  //
  /**
   * @brief n_faces
   */
  const unsigned n_faces;

  //
  //
  //  unsigned poly_order;

  //  unsigned n_face_bases, n_cell_bases;

  //
  //
  /**
   * @brief id_num
   */
  unsigned id_num;

  //
  //
  /**
   * We want to know which degrees of freedom are restrained and which are open.
   * Hence, we store a bitset which has its size equal to the number of dofs of
   * each face of the cell and it is 1 if the dof is open, and 0 if it is
   * restrained.
   */
  std::vector<boost::dynamic_bitset<> > dof_names_on_faces;

  //
  //
  /**
   * @brief assign_local_global_cell_data
   */
  void assign_local_global_cell_data(const unsigned &i_face,
                                     const unsigned &local_num_,
                                     const unsigned &global_num_,
                                     const unsigned &comm_rank_,
                                     const unsigned &half_range_);

  //
  //
  /**
   * @brief assign_local_cell_data
   */
  void assign_local_cell_data(const unsigned &i_face,
                              const unsigned &local_num_,
                              const int &comm_rank_,
                              const unsigned &half_range_);

  //
  //
  /**
   * @brief assign_ghost_cell_data
   */
  void assign_ghost_cell_data(const unsigned &i_face,
                              const int &local_num_,
                              const int &global_num_,
                              const unsigned &comm_rank_,
                              const unsigned &half_range_);

  //
  //
  /**
   * A unique ID of each cell, which is taken from the dealii cell
   * corresponding to the current cell. This ID is unique in the
   * interCPU space.
   */
  std::string cell_id;

  //
  //
  /**
   * @brief Decides if the current face has a coarser neighbor.
   */
  std::vector<unsigned> half_range_flag;

  //
  //
  /**
   * @brief The CPU number of the processor which owns the current face.
   */
  std::vector<unsigned> face_owner_rank;

  //
  //
  /**
   * @brief An iterator to the deal.II element corresponding to this Cell.
   */
  dealii_cell_type dealii_cell;

  //
  //
  /**
   * @brief dofs_ID_in_this_rank
   */
  std::vector<std::vector<int> > dofs_ID_in_this_rank;

  //
  //
  /**
   * @brief dofs_ID_in_all_ranks
   */
  std::vector<std::vector<int> > dofs_ID_in_all_ranks;

  //
  //
  /**
   * @brief Contains all of the boundary conditions of on the faces of this
   * Cell.
   */
  std::vector<BC> BCs;

  //
  //
  /**
   * @brief A pointer to the BaseModel object which contains this Cell.
   */
  BaseModel *model;
};

//
//
/**
 * This the diffusion model problem. The original method of solving this is
 * based on hybridized DG.
 * \ingroup modelelements
 */
template <int dim, int spacedim = dim>
struct Diffusion : public Cell<dim, spacedim>
{
  //
  //
  /**
   * @brief We use the same typename as we defined in base class.
   */
  using typename Cell<dim, spacedim>::dealii_cell_type;

  //
  //
  /**
   * @brief The constructor of the class.
   */
  Diffusion(dealii_cell_type &inp_cell,
            const unsigned id_num_,
            BaseModel *model_);

  //
  //
  /**
   * @brief The destructor of the class.
   */
  ~Diffusion() {}
};
}

#include "../../source/elements/cell.cpp"
#endif // CELL_CLASS_HPP
