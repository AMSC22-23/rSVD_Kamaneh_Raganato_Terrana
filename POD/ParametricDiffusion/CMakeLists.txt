cmake_minimum_required(VERSION 3.12.0)
project(ParametricDiffusion LANGUAGES CXX C)

# https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html
# find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include(../common/cmake-common.cmake)

# add_executable(ParametricDiffusion src/ParametricDiffusion.cpp src/POD.cpp src/Poisson2D.cpp)
# deal_ii_setup_target(ParametricDiffusion)
# target_link_libraries (ParametricDiffusion Eigen3::Eigen)

add_executable(ParametricDiffusion_snapshot src/ParametricDiffusion_snapshot.cpp src/POD.cpp src/Poisson2D_snapshot.cpp)
deal_ii_setup_target(ParametricDiffusion_snapshot)
# target_link_libraries (ParametricDiffusion_snapshot Eigen3::Eigen)