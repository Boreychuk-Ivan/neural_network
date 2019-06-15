cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(neural_network_lib)

set(SOURCES src/neuron.cpp>
            src/neural_network.cpp>
            src/layer.cpp>
            src/back_propagation.cpp>
            src/training.cpp>
            src/file_handler.cpp>
            src/loss_functions.cpp>
            src/error_handle.cpp
            )

set(HEADERS include/neuron.h>
            include/neural_network.h>
            include/layer.h>
            include/activation_functions.h>
            include/back_propagation.h>
            include/training.h>
            include/file_handler.h>
            include/loss_functions.h>
            include/error_handle.h>
            )

add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME}
      PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(neural_network_lib PUBLIC pch)