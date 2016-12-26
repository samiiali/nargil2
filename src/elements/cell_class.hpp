#include <type_traits>

#include <deal.II/fe/fe_values.h>

#ifndef CELL_CLASS_HPP
#define CELL_CLASS_HPP

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

/*!
 * \defgroup cells Cell data
 * \brief
 * This group contains the classes which encapsulate data corresponding to each
 * cell in the mesh.
 */

/*!
 * \brief The \c Cell_Class contains most of the required data about a generic
 * element in the mesh.
 *
 * \ingroup cells
 */
template <int dim, int spacedim = dim> struct Cell
{
  typedef dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim> >
    dealiiCell;
  typedef
    typename std::vector<std::unique_ptr<Cell> >::iterator vec_iter_ptr_type;

  typedef std::unique_ptr<dealii::FEValues<dim> > FE_val_ptr;
  typedef std::unique_ptr<dealii::FEFaceValues<dim> > FEFace_val_ptr;

  /**
   * \details
   * We remove the default constructor to avoid uninitialized creation of Cell
   * objects.
   */
  Cell() = delete;
  /**
   * \details
   * The constructor of this class takes a deal.II cell and its deal.II ID.
   * \param inp_cell The iterator to the deal.II cell in the mesh.
   * \param id_num_  The unique ID (\c dealii_Cell::id()) of the dealii_Cell.
   * This is necessary when working on a distributed mesh.
   */
  Cell(dealiiCell &inp_cell,
       const unsigned &id_num_,
       const unsigned &poly_order_);

  /**
   * \details
   * We remove the copy constructor of this class to avoid unnecessary copies
   * (specially unintentional ones). Up to October 2015, this copy constructor
   * was not useful anywhere in the code.
   */

  Cell(const Cell &inp_cell) = delete;

  /**
   * \details
   * We need a move constructor, to be able to pass this class as function
   * arguments efficiently. Maybe, you say that this does not help efficiency
   * that much, but we are using it for semantic constraints.
   * \param inp_cell An object of the \c Cell_Class type which we steal its
   * guts.
   */
  Cell(Cell &&inp_cell) noexcept;

  /**
   * Obviously, the destructor.
   */
  virtual ~Cell();

  /**
   * \brief Moves the dealii::FEValues and dealii::FEFaceValues
   * objects between different elements and faces.
   *
   * dealii::FEValues are not copyable objects. They also
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
   *
   * \details We attach a \c unique_ptr of dealii::FEValues and
   * dealii::FEFaceValues to the current object.
   * \param cell_quad_fe_vals_ The dealii::FEValues which is used for location
   * of quadrature points in cells.
   * \param face_quad_fe_vals_ The dealii::FEValues which is used for lacation
   * of support points in cells.
   * \param cell_supp_fe_vals_ The dealii::FEValues which is used for location
   * of quadrature points on faces.
   * \param face_supp_fe_vals_ The dealii::FEValues which is used for location
   * of support points on faces.
   */
  void attach_FEValues(FE_val_ptr &cell_quad_fe_vals_,
                       FEFace_val_ptr &face_quad_fe_vals_,
                       FE_val_ptr &cell_supp_fe_vals_,
                       FEFace_val_ptr &face_supp_fe_vals_);

  void detach_FEValues(FE_val_ptr &cell_quad_fe_vals_,
                       FEFace_val_ptr &face_quad_fe_vals_,
                       FE_val_ptr &cell_supp_fe_vals_,
                       FEFace_val_ptr &face_supp_fe_vals_);

  /*!
   * \details Updates the FEValues which are connected to the current element
   * (not the FEFaceValues.)
   */
  void reinit_cell_fe_vals();

  /*!
   * \details Updates the FEFaceValues which are connected to a given face of
   * the current element.
   * \param i_face the face which we want to update the connected FEFaceValues.
   * \c i_face\f$\in\{1,2,3,4\}\f$
   */
  void reinit_face_fe_vals(unsigned i_face);

  const unsigned n_faces;
  unsigned poly_order, n_face_bases, n_cell_bases;
  unsigned id_num;

  /**
   * We want to know which degrees of freedom are restrained and which are open.
   * Hence, we store a bitset which has its size equal to the number of dofs of
   * each face of the cell and it is 1 if the dof is open, and 0 if it is
   * restrained.
   */
  std::vector<boost::dynamic_bitset<> > dof_names_on_faces;

  void assign_local_global_cell_data(const unsigned &i_face,
                                     const unsigned &local_num_,
                                     const unsigned &global_num_,
                                     const unsigned &comm_rank_,
                                     const unsigned &half_range_);

  void assign_local_cell_data(const unsigned &i_face,
                              const unsigned &local_num_,
                              const int &comm_rank_,
                              const unsigned &half_range_);

  void assign_ghost_cell_data(const unsigned &i_face,
                              const int &local_num_,
                              const int &global_num_,
                              const unsigned &comm_rank_,
                              const unsigned &half_range_);

  /**
   * A unique ID of each cell, which is taken from the dealii cell
   * corresponding to the current cell. This ID is unique in the
   * interCPU space.
   */
  std::string cell_id;

  std::vector<unsigned> half_range_flag;
  std::vector<unsigned> face_owner_rank;
  dealiiCell dealii_cell;
  std::vector<std::vector<int> > dofs_ID_in_this_rank;
  std::vector<std::vector<int> > dofs_ID_in_all_ranks;
  std::vector<BC> BCs;

  std::unique_ptr<dealii::FEValues<dim> > cell_quad_fe_vals, cell_supp_fe_vals;
  std::unique_ptr<dealii::FEFaceValues<dim> > face_quad_fe_vals,
    face_supp_fe_vals;

  /**
   *
   */
  //  const dealii::QGauss<dim> *elem_quad_bundle;

  /**
   *
   */
  //  const dealii::QGauss<dim - 1> *face_quad_bundle;
};

#include "cell_class.cpp"
#endif // CELL_CLASS_HPP
