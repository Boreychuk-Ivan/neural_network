cmake_minimum_required(VERSION 3.0.0)

project(tests)

# Link googletest
add_subdirectory(googletest)

link_directories(${CMAKE_SOURCE_DIR}/neural_network_lib)

include_directories(${CMAKE_SOURCE_DIR}/neural_network_lib)

include_directories(${CMAKE_SOURCE_DIR}/tests)

set (SOURCES matrix_tests.h>
             neuron_tests.h>
             layer_tests.h>
             neural_network_tests.h>
             file_handler_tests.h>
             back_propagation_tests.h>
             run_all_tests.cpp>
             auxillary_functions.h>
             )

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} neural_network_lib gtest)

