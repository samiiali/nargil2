#ifndef MODEL_OPTIONS
#define MODEL_OPTIONS

namespace nargil
{
namespace model_options
{
/**
 * This enum includes all model options. Obviously, some of these options
 * cannot be used together; e.g. a model cannot have both implicit and
 * explicit time integration.
 * @todo Write functions which are asserting the consistency of the model
 * options assigned by the user.
 */
enum options
{
  implicit_time_integration = 1 << 0,
  explicit_time_integration = 1 << 1,
  CG_dof_numbering = 1 << 6,
  DG_dof_numbering = 1 << 7,
  hybridized_dof_numbering = 1 << 8
};
}

namespace bases_options
{

//
//
/**
 * @brief The BasesOptions enum
 */
enum options
{
  HDG = 1 << 0,
  nodal = 1 << 3,
  modal = 1 << 4,
  polynomial = 1 << 5
};
}

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
}

#endif
