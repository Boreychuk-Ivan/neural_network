cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(main)

set(SOURCES main.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} matrix_lib neural_network_lib)

