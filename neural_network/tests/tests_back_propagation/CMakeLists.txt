cmake_minimum_required(VERSION 3.0.0)

project(tests_back_propagation)

link_directories(${CMAKE_SOURCE_DIR}/neural_network_lib)

include_directories(${CMAKE_SOURCE_DIR}/neural_network_lib)

set (SOURCE tests_back_propagation.cpp ../auxillary_functions.h)

add_executable(${PROJECT_NAME} ${SOURCE})

target_link_libraries(${PROJECT_NAME} neural_network_lib gtest)

