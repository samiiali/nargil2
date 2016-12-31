#include <assert.h>
#include <memory>
#include <type_traits>

#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#ifndef CELL_CLASS_HPP
#define CELL_CLASS_HPP

#include "../models/dof_numbering.hpp"
#include "../models/model_options.hpp"

/**
 * \defgroup modelelements Model Elements
 * \brief This group contains different model elements and the relevant
 * structures used to solve different model problems.
 */

namespace nargil
{
//
//
//
namespace bases
{
//
//
/**
 *
 */
template <int dim, int spacedim> struct basis
{
  basis() {}
  ~basis() {}
  template <typename BasisType> std::unique_ptr<BasisType> create();
};

//
//
/**
 *
 */
template <int dim, int spacedim = dim>
struct hdg_diffusion_polybasis : public basis<dim, spacedim>
{
  hdg_diffusion_polybasis() = delete;
  hdg_diffusion_polybasis(const unsigned poly_order, const unsigned quad_order);

  unsigned _poly_order;
  unsigned _quad_order;

  dealii::FE_DGQ<dim> u_basis;
  dealii::FESystem<dim> q_basis;
  dealii::FE_DGQ<dim - 1> uhat_basis;

  dealii::QGauss<dim> cell_quad;
  dealii::QGauss<dim - 1> face_quad;

  dealii::Quadrature<dim> projected_face_quad;

  dealii::UpdateFlags fe_val_flags;

  dealii::FEValues<dim> u_in_cell;
  dealii::FEValues<dim> q_in_cell;
  dealii::FEFaceValues<dim> uhat_on_face;
  std::vector<std::unique_ptr<dealii::FEValues<dim> > > u_on_faces;
  std::vector<std::unique_ptr<dealii::FEValues<dim> > > q_on_faces;

  //
  //
  /**
   *
   */
  void adjusted_subface_quad_points(const dealii::Point<dim - 1> &P0,
                                    const unsigned half_range);

  //
  //
  /**
   *
   */
  unsigned get_n_dofs_on_each_face();

  //
  //
  /**
   *
   */
  static nargil::bases::basis_options get_options();
};
}

//
//
// Forward decleration of BaseModel. The model, which all other models
// are based on it.
struct base_model;

//
//
/**
 * \brief The enum which contains different boundary conditions.
 */
enum class boundary_condition
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
template <int dim, int spacedim = dim> struct cell
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
   * \details
   * We remove the default constructor to avoid uninitialized creation of Cell
   * objects.
   */
  cell() = delete;

  //
  //
  /**
   * \details
   * The constructor of this class takes a deal.II cell and creates the cell.
   */
  cell(dealii_cell_type &inp_cell, const unsigned id_num_, base_model *model_);

  //
  //
  /**
   * \details
   * We remove the copy constructor of this class to avoid unnecessary copies
   * (specially unintentional ones).
   */
  cell(const cell &inp_cell) = delete;

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
  cell(cell &&inp_cell) noexcept;

  //
  //
  /**
   * @brief The destructor of the class.
   */
  virtual ~cell();

  //
  //
  /**
   * @brief This is the factory function which creates a cell of type
   * ModelEq (the template parameter). This function is called by
   * Model::init_mesh_containers.
   */
  template <typename ModelEq, typename BasisType>
  static std::unique_ptr<ModelEq> create(dealii_cell_type &inp_cell,
                                         const unsigned id_num_,
                                         BasisType *,
                                         base_model *model_);

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
  std::vector<boundary_condition> BCs;

  //
  //
  /**
   * @brief A pointer to the BaseModel object which contains this Cell.
   */
  base_model *my_model;
};

//
//
//
//
/**
 * This the diffusion model problem. The original method of solving this is
 * based on hybridized DG.
 * \ingroup modelelements
 */
template <int dim, int spacedim = dim>
struct diffusion_cell : public cell<dim, spacedim>
{
  //
  //
  /**
   * @brief We use the same typename as we defined in base class.
   */
  using typename cell<dim, spacedim>::dealii_cell_type;

  //
  //
  /**
   * @brief The constructor of the class.
   */
  diffusion_cell(dealii_cell_type &inp_cell,
                 const unsigned id_num_,
                 bases::basis<dim, spacedim> *basis,
                 base_model *model_,
                 bases::basis_options basis_opts);

  //
  //
  /**
   * @brief The destructor of the class.
   */
  ~diffusion_cell() {}

  //
  //
  /**
   *
   */
  template <typename Func> void assign_BCs(Func f);

  //
  //
  /**
   *
   */
  void assemble_globals();

  //
  //
  /*
   *
   */
  unsigned get_relevant_dofs_count(const unsigned);

  //
  //
  /**
   * @brief my_basis
   */
  bases::basis<dim, spacedim> *my_basis;

  //
  //
  /**
   *
   */
  bases::basis_options my_basis_opts;
};

//
//
//
//
/**
 *
 */
}

#include "../../source/elements/cell.cpp"
#endif // CELL_CLASS_HPP
