#include <string>
#include <vector>

#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/parallel_vector.h>

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

namespace LA
{
using namespace dealii::LinearAlgebraPETSc;
}

namespace nargil
{
/**
 * This is a simple tokenizer function. I could not find a better
 * place to put it.
 */
void Tokenize(const std::string &str_in,
              std::vector<std::string> &tokens,
              const std::string &delimiters);

//
//
//
//
//

template <typename T> void reck_it_Ralph(T *obj) { T().swap(*obj); }

//
//
//
//
//

template <int dim, int spacedim = dim> struct distributed_vector
{
  distributed_vector(const dealii::DoFHandler<dim, spacedim> &dof_handler,
                     const MPI_Comm &comm)
    : my_dof_handler(&dof_handler),
      idx_set(my_dof_handler->locally_owned_dofs()),
      my_comm(&comm),
      local_nodal_vec(idx_set, *my_comm)
  {
    idx_set.fill_index_vector(idx_vec);
  }

  void assemble(const unsigned &idx, const double &val)
  {
    local_nodal_vec[idx_vec[idx]] = val;
  }

  void reinit_global_vec(LA::MPI::Vector &global_nodal_vec)
  {
    dealii::IndexSet active_idx_set;
    dealii::DoFTools::extract_locally_relevant_dofs(*my_dof_handler,
                                                    active_idx_set);
    global_nodal_vec.reinit(idx_set, active_idx_set, *my_comm);
  }

  void copy_to_global_vec(LA::MPI::Vector &global_nodal_vec,
                          const bool &do_reinit_global_vec = true)
  {
    if (do_reinit_global_vec)
      reinit_global_vec(global_nodal_vec);
    local_nodal_vec.compress(dealii::VectorOperation::insert);
    global_nodal_vec = local_nodal_vec;
  }

  const dealii::DoFHandler<dim> *const my_dof_handler;
  dealii::IndexSet idx_set;
  std::vector<unsigned> idx_vec;
  const MPI_Comm *const my_comm;
  LA::MPI::Vector local_nodal_vec;
};
}

#include "../../source/misc/utils.cpp"

#endif
