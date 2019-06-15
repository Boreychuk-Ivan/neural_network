cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(neural_network_project)

#Set C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)

#Set debug flug
string(APPEND CMAKE_C_FLAGS_DEBUG " -DDEBUG_FLAG")
string(APPEND CMAKE_CXX_FLAGS_DEBUG " -DDEBUG_FLAG")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MTd")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")

#Directories
add_subdirectory(neural_network_lib)
add_subdirectory(matrix_lib)
add_subdirectory(project)
add_subdirectory(examples)

#PCH
set(PCH_SOURCES pch.cpp pch.h)
add_library(pch STATIC ${PCH_SOURCES})
target_include_directories(pch
      PUBLIC ${PROJECT_SOURCE_DIR})