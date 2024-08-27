cmake_minimum_required(VERSION 3.12.0)
project(ParametricDiffusion LANGUAGES CXX C)

# https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html
# find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include(../common/cmake-common.cmake)

add_executable(Diff1D_A_one_prm src/Diff1D_A_one_prm.cpp src/POD.cpp src/AdvDiff1D.cpp src/AdvDiff1D_POD.cpp)
deal_ii_setup_target(Diff1D_A_one_prm)

# add_executable(Diff1D_B_more_prm src/Diff1D_B_more_prm.cpp src/POD.cpp src/AdvDiff1D.cpp src/AdvDiff1D_POD.cpp)
# deal_ii_setup_target(Diff1D_B_more_prm)

# add_executable(Diff1D_C_new_prm src/Diff1D_C_new_prm.cpp src/POD.cpp src/AdvDiff1D.cpp src/AdvDiff1D_POD.cpp)
# deal_ii_setup_target(Diff1D_C_new_prm)

# add_executable(Diff1D_C_new_prm src/Diff1D_C_new_prm.cpp ../../PCA/src/SVD_class.cpp ../../PCA/src/PM.cpp 
# ../../PCA/src/Jacobi_Class.cpp ../../PCA/src/JacobiOperations.cpp src/POD.cpp src/AdvDiff1D.cpp src/AdvDiff1D_POD.cpp)
# deal_ii_setup_target(Diff1D_C_new_prm)

# add_executable(Diff1D_D_openmp src/Diff1D_D_openmp.cpp src/POD.cpp src/AdvDiff1D.cpp src/AdvDiff1D_POD.cpp)
# deal_ii_setup_target(Diff1D_D_openmp)

# target_link_libraries (ParametricDiffusion1D Eigen3::Eigen)