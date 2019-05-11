cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(libarary_2)

set(SOURCES libarary_2.cpp)
set(HEADERS libarary_2.h)

add_library(libarary_2 STATIC ${SOURCES} ${HEADERS})

target_include_directories(libarary_2 PUBLIC ${PROJECT_SOURCE_DIR})

target_include_directories(libarary_2 PUBLIC ../src/libarary_2)