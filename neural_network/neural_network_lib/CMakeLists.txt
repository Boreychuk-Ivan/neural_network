cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(neural_network_lib)

set(SOURCES src/neuron.cpp>
            src/neural_network.cpp>
            src/layer.cpp)

set(HEADERS include/neuron.h>
            include/neural_network.h>
            include/layer.h)

add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME}
      PUBLIC ${PROJECT_SOURCE_DIR}/include)