cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(neural_network_lib)

set(SOURCES neural_network_lib.cpp)
set(HEADERS neural_network_lib.h)

add_library(neural_network_lib STATIC ${SOURCES} ${HEADERS})

target_include_directories(neural_network_lib PUBLIC ${PROJECT_SOURCE_DIR})

target_include_directories(neural_network_lib PUBLIC ../src/neural_network_lib)