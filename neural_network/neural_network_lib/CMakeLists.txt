cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(neural_network_lib)

set(SOURCES src/neuron.cpp>
            src/neural_network.cpp>
            src/layer.cpp>
            src/input_layer.cpp>
            src/back_propagation.cpp>
            src/forward_propagation.cpp)

set(HEADERS include/neuron.h>
            include/neural_network.h>
            include/layer.h>
            include/input_layer.h>
            include/back_propagation.h>
            include/forward_propagation.h>
            include/activation_functions.h)

add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME}
      PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(neural_network_lib PUBLIC pch)