cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

set(LIBRARY_1 matrix_lib)
set(LIBRARY_2 neural_network_lib)

add_subdirectory(${LIBRARY_1})
add_subdirectory(${LIBRARY_2})

