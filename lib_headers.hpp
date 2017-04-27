//#define EIGEN_USE_MKL_ALL

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <memory>
#include <unistd.h>

#include <mpi.h>
#include <petscis.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>
#include <slepc.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/vector.h>

#define USE_PETSC_LA
namespace LA
{
#ifdef USE_PETSC_LA
using namespace dealii::LinearAlgebraPETSc;
#else
using namespace ::LinearAlgebraTrilinos;
#endif
}
#include <deal.II/lac/petsc_parallel_vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/base/derivative_form.h>
#include <deal.II/base/function.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/polynomials_abf.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/Sparse>

#include <boost/dynamic_bitset.hpp>
#include <boost/numeric/mtl/mtl.hpp>

#ifndef LIB_HEADERS_HPP
#define LIB_HEADERS_HPP

// typedef Eigen::MatrixXd eigen3mat;
// typedef Eigen::SparseMatrix<double> eigen3sparse_mat;
// typedef Eigen::Triplet<double> eigen3triplet;
// typedef Eigen::LDLT<eigen3mat, Eigen::Lower> eigen3ldlt;

// template <int dim, template <int> class CellType>
// struct hdg_model;

// template <int dim, template <int> class CellType>
// struct explicit_hdg_model;

// template <int dim, template <int> class CellType>
// struct hdg_model_with_explicit_rk;

#endif
