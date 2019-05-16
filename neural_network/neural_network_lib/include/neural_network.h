#pragma once 

#include "pch.h"
#include "layer.h"


class NeuralNetwork
{
private:
    std::vector<Layer> m_neuron_layers;
public:
    NeuralNetwork() = delete;
    NeuralNetwork(const std::vector<unsigned>& kArchitecture);
    
    void PushInputData(const Vector<double> kInputVector);
    void CalculateOutputs();

    void SetActivationFunction(const unsigned& kNumLayer, const ActivationFunctionType& kActivationFunction);

    Vector<double> GetOutputs();

    void DisplayArchitecture();
    void DisplayLayerParametrs(const unsigned& kNumLayer);
    void DisplayLayerNeurons(const unsigned& kNumLayer);
};

