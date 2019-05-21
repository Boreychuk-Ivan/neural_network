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
    Vector<double> CalculateOutputs();
    void CalculateDerivativeValues();
    Vector<double> CalculateDerivativeValuesOnLayer(const unsigned& kNumLayer);

    void SetActivationFunction(const unsigned& kNumLayer, const ActivationFunctionType& kActivationFunction);
    void SetSynapticWeigths(const unsigned& kNumLayer, const Matrix<double> kSynapticWeigths);
    void SetBiases(const unsigned& kNumLayer, const Vector<double> kBiases);

    Vector<double> GetOutputs();
    Vector<double> GetDerivativeValues(const unsigned& kNumLayer);

    void DisplayArchitecture();
    void DisplayLayers();
    void DisplayNeurons();
    void DisplayLayerParametrs(const unsigned& kNumLayer);
    void DisplayLayerNeurons(const unsigned& kNumLayer);
};

