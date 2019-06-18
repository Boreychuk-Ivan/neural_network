cmake_minimum_required(VERSION 3.0.0)

project(tests_neuron)

link_directories(${CMAKE_SOURCE_DIR}/neural_network_lib)

include_directories(${CMAKE_SOURCE_DIR}/neural_network_lib)

set (SOURCE tests_neuron.cpp ../auxillary_functions.h)

add_executable(${PROJECT_NAME} ${SOURCE})

target_link_libraries(${PROJECT_NAME} neural_network_lib gtest)

