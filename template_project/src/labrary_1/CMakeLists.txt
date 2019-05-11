cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(libarary_1)

set(SOURCES libarary_1.cpp)
set(HEADERS libarary_1.h)

add_library(libarary_1 STATIC ${SOURCES} ${HEADERS})

target_include_directories(libarary_1 PUBLIC ${PROJECT_SOURCE_DIR})

target_include_directories(libarary_1 PUBLIC ../src/libarary_1)