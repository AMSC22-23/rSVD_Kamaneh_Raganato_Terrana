cmake_minimum_required(VERSION 3.12.0)
project(ParametricDiffusion LANGUAGES CXX C)

# https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html
# find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include(../common/cmake-common.cmake)

add_executable(Diff1D src/Diff1D.cpp ../../src/PM.cpp ../../src/Jacobi_Class.cpp ../../src/JacobiOperations.cpp
src/POD.cpp src/AdvDiff1D.cpp src/AdvDiff1D_POD.cpp)
deal_ii_setup_target(Diff1D)

# add_executable(Diff1D_new_prm src/Diff1D_new_prm.cpp ../../src/PM.cpp ../../src/Jacobi_Class.cpp ../../src/JacobiOperations.cpp
# src/POD.cpp src/AdvDiff1D.cpp src/AdvDiff1D_POD.cpp)
# deal_ii_setup_target(Diff1D_new_prm)

# add_executable(Diff1D_convergence src/Diff1D_convergence.cpp ../../src/PM.cpp ../../src/Jacobi_Class.cpp ../../src/JacobiOperations.cpp
# src/POD.cpp src/AdvDiff1D.cpp src/AdvDiff1D_POD.cpp)
# deal_ii_setup_target(Diff1D_convergence)