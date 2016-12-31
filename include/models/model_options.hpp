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
  LDG_dof_numbering = 1 << 7,
  HDG_dof_numbering = 1 << 8
};
}

namespace bases
{
//
//
/**
 * @brief The BasesOptions enum
 */
enum basis_options
{
  HDG = 1 << 0,
  LDG = 1 << 1,
  nodal = 1 << 2,
  modal = 1 << 3,
  polynomial = 1 << 4
};
}
}

#endif
