cmake_minimum_required(VERSION 3.0)

project(logical_xor)

set(SOURCES main.cpp)

include_directories(./data)

FILE(COPY ./data DESTINATION ${PROJECT_BINARY_DIR})

link_directories(${CMAKE_SOURCE_DIR}/neural_network_lib)
include_directories(${CMAKE_SOURCE_DIR}/neural_network_lib)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} neural_network_lib)