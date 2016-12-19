#include <cstdio>
#include <fstream>
#include <iostream>
#include <slepc.h>

#include "src/models/generic_model.hpp"

int main(int argc, char **argv)
{
  SlepcInitialize(&argc, &argv, (char *)0, NULL);
  PetscMPIInt comm_rank, comm_size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &comm_rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &comm_size);
  // dealii::MultithreadInfo::set_thread_limit(1);

  int number_of_threads = 1;
#ifdef _OPENMP
  omp_set_num_threads(1);
  number_of_threads = omp_get_max_threads();
#endif

  if (comm_rank == 0)
  {
    char help_line[300];
    std::snprintf(help_line,
                  300,
                  "\n"
                  "mpiexec -n 6 ./GN_Solver -h_0 2 -h_n 8 -p_0 1 -p_n 2 -amr 0 "
                  "\n");
    std::cout << "Usage: " << help_line << std::endl;

    std::ofstream Convergence_Cleaner("Convergence_Result.txt");
    Convergence_Cleaner.close();
    std::ofstream ExecTime_Cleaner("Execution_Time.txt");
    ExecTime_Cleaner.close();
  }

  int p_1, p_2, h_1, h_2, found_options = 1;
  PetscBool found_option;
  int adaptive_on = 0;

  PetscOptionsGetInt(NULL, "-p_0", &p_1, &found_option);
  found_options = found_option * found_options;
  PetscOptionsGetInt(NULL, "-p_n", &p_2, &found_option);
  found_options = found_option && found_options;
  PetscOptionsGetInt(NULL, "-h_0", &h_1, &found_option);
  found_options = found_option && found_options;
  PetscOptionsGetInt(NULL, "-h_n", &h_2, &found_option);
  found_options = found_option && found_options;
  PetscOptionsGetInt(NULL, "-amr", &adaptive_on, &found_option);
  found_options = found_option && found_options;

  const int dim = 2;

  //  for (unsigned p1 = (unsigned)p_1; p1 < (unsigned)p_2; ++p1)
  //  {
  //    SolutionManager<dim> sol_man1(p1,
  //                                  PETSC_COMM_WORLD,
  //                                  comm_size,
  //                                  comm_rank,
  //                                  number_of_threads,
  //                                  adaptive_on);
  //    sol_man1.solve((unsigned)h_1, (unsigned)h_2);
  //  }

  SlepcFinalize();
  return 0;
}
