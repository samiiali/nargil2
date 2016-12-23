# Set the name of the project and target:
set(TARGET "nargil")

include(CMakeLocals/CMakeIncs.txt)

# IMPORTANT : THIS PART CORRESPONDS TO GCC COMPILER. HOWEVER YOU CAN USE
# GCC WITH INTEL libiomp5.so
# set(MKLROOT "/opt/apps/sysnet/intel/12.1/mkl/10.3.12.361")
# set(COMPILER_LIB "/opt/apps/ossw/applications/gcc/gcc-4.8/sl6/lib64")

# IMPORTANT : THIS PART IS FOR INTEL COMPILER
#set(MKLROOT "/opt/apps/sysnet/intel/12.1/mkl/10.3.12.361")
#set(COMPILER_LIB "/opt/apps/sysnet/intel/12.1/lib/intel64")

# include_directories(${INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR} "${MKLROOT}/include" ${EIGEN_INC})
# link_directories(${LINK_DIRECTORIES} "${MKLROOT}/lib/intel64" ${COMPILER_LIB})

# Declare all source files the target consimts of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#    FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#    FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#    SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.

SET(TARGET_SRC src/main.cpp)

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.2 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Are all dependencies fullfilled?
#
IF(NOT DEAL_II_WITH_PETSC OR NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_PETSC = ON
    DEAL_II_WITH_P4EST = ON
One or all of these are OFF in your installation but are required for this tutorial step."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET} CXX)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  message("Compiler is Clang")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  message("Compiler is GNU")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  message("Compiler is Intel")
endif()

DEAL_II_INVOKE_AUTOPILOT()

#
# We want to include all of the pure header files, i.e. class templates
# in the source tree
#
# add_subdirectory(src/poly_bases)
add_subdirectory(src/models)
add_subdirectory(src/solution_managers)
add_subdirectory(src/elements)

# add_custom_target(DUMMY SOURCES
#                   src/solution_manager.hpp
#                   src/solution_manager.cpp
#                   src/lib_headers.hpp
#                   src/support_classes.hpp
#                   src/support_classes.cpp
#                   src/solvers.hpp
#                   src/solvers.cpp
#                   src/time_integrators.hpp
#                   src/time_integrators.cpp)

include(CMakeLocals/CMakeFlags.txt)

#set(CMAKE_CXX_FLAGS " -O2 -openmp -std=c++0x ${CMAKE_CXX_FLAGS}")
#target_link_libraries(${TARGET} mkl_intel_lp64 mkl_core iomp5 mkl_intel_thread pthread m)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
