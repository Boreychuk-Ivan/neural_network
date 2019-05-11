cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(main)

set(SOURCES main.cpp)

add_executable(main ${SOURCES})

target_include_directories(main PUBLIC ../src/libarary_1)
target_include_directories(main PUBLIC ../src/libarary_2)

target_link_libraries(main PUBLIC libarary_1)
target_link_libraries(main PUBLIC libarary_2)
