#pragma once

#include "pch.h"
#include "back_propagation.h"
#include "neural_network.h"

class Training
{
public:
    //Constructors
    Training() = delete;
    Training
    (
        NeuralNetwork& kNeuralNetwork, 
        const double kLearningRate, 
        const double kMomentum, 
        const std::string& kFilePath, 
        const size_t kInputsNumber, 
        const size_t kOutputsNumber, 
        const size_t kTrainingSetSize
    );

    //Getters
    size_t GetInputsNumber() const;
    size_t GetOutputsNumber() const;
    size_t GetTrainingSetSize() const;
    Matrix<double> GetInputMatrix() const;
    Matrix<double> GetOutputMatrix() const;


    //Methods
    void ReadFile();
    void TrainOnSet();
    void TrainNeuralNetwork(const size_t& kEpochNumber);


private:
    BackPropagation m_neural_network;
    std::ifstream m_training_file;

    size_t m_inputs_number;
    size_t m_outputs_number;
    size_t m_training_set_size;

    Matrix<double> m_input_matrix;
    Matrix<double> m_targets_matrix;
};