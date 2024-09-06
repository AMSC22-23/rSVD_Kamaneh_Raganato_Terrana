cmake_minimum_required(VERSION 3.12)

project(YourProjectName)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -Wextra")

# Set the path to Eigen headers
set(EIGEN_INCLUDE_DIR "/path/to/eigen")

# Source files
file(GLOB_RECURSE SRC_FILES "src/*.cpp")
list(FILTER SRC_FILES EXCLUDE REGEX "src/exclude_file.cpp")

# Test source files
file(GLOB_RECURSE TEST_SRC_FILES "tests/*.cpp")
list(FILTER TEST_SRC_FILES EXCLUDE REGEX "tests/exclude_test.cpp")

# Include directories
include_directories(include ${EIGEN_INCLUDE_DIR})

# Executable for the main program
add_executable(YourProgram ${SRC_FILES})

# Executables for tests
foreach(TEST_SRC ${TEST_SRC_FILES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SRC} ${SRC_FILES})
endforeach()

# Target to build and run tests without profiling
add_custom_target(tests)
foreach(TEST_NAME ${TEST_SRC_FILES})
    get_filename_component(TEST_EXEC_NAME ${TEST_NAME} NAME_WE)
    add_custom_command(TARGET tests POST_BUILD COMMAND ${TEST_EXEC_NAME})
endforeach()

# Target to build and run tests with profiling
add_custom_target(profile COMMAND ${CMAKE_COMMAND} --build . --target tests)
add_custom_command(TARGET profile POST_BUILD COMMAND gprof YourProgram > profile_output.txt)

# Clean up generated files and remove files inside /data/output/
add_custom_target(clean-all)
add_custom_command(TARGET clean-all COMMAND rm -rf ${CMAKE_BINARY_DIR}/*)
