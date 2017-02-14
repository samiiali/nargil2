#ifndef MODEL_OPTIONS
#define MODEL_OPTIONS

/**
 * @defgroup all_enums Enumerations
 */

namespace nargil
{

/**
 * @ingroup all_enums
 *
 * The enum which contains different boundary conditions.
 *
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
