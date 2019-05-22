#pragma once

#include "pch.h"
#include "back_propagation.h"
#include "neural_network.h"

class Training
{
private:
    BackPropagation m_neural_network;
    std::ifstream m_training_file;
    
    size_t m_inputs_number;
    size_t m_outputs_number;
    size_t m_training_set_size;

    Matrix<double> m_input_matrix;
    Matrix<double> m_output_matrix;
public:
    //Constructors
    Training() = delete;
    Training
    (
        const NeuralNetwork& kNeuralNetwork, 
        const double kLearningRate, 
        const double kMomentum, 
        const std::string& kFilePath, 
        const size_t kInputsNumber, 
        const size_t kOutputsNumber, 
        const size_t kTrainingSetSize
    );


    //Getters
    Matrix<double> GetInputMatrix();
    Matrix<double> GetOutputMatrix();

    //Methods
    void ReadFile();
    void TrainOnSet();
    void TrainNeuralNetwork();
};