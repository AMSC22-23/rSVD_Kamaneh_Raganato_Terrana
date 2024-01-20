cmake_minimum_required(VERSION 3.12.0)
project(ParametricDiffusion LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(ParametricDiffusion src/ParametricDiffusion.cpp src/Poisson2D.cpp)
deal_ii_setup_target(ParametricDiffusion)

add_executable(ParametricDiffusion_snapshot src/ParametricDiffusion_snapshot.cpp src/Poisson2D_snapshot.cpp)
deal_ii_setup_target(ParametricDiffusion_snapshot)