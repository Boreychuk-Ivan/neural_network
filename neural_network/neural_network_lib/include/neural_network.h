#pragma once 

#include "pch.h"
#include "layer.h"


class NeuralNetwork
{
public:
    NeuralNetwork() = delete;
    NeuralNetwork(const std::vector<int>& kArchitecture);
    
    Vector<double> CalculateOutputs(const Vector<double> kInputVector);
	Vector<double> FeedForward(const Vector<double> kInputVector);		//Calculate outputs + derivative values
	
    //Setters
    void SetActivationFunction(const size_t& kNumLayer, const ActivationFunctionType& kActivationFunction);
    void SetSynapticWeigths(const size_t& kNumLayer, const Matrix<double> kSynapticWeigths);
    void SetBiases(const size_t& kNumLayer, const Vector<double> kBiases);

    //Getters
    Vector<double> GetOutputs() const;
    Layer GetLayer(const size_t& kNumLayer) const;
    Layer& GetLayer(const size_t& kNumLayer);

    size_t GetLayersNumber() const;
    size_t GetInputsNumber() const;
    size_t GetOutputsNumber() const;

    //Displays
    void DisplayArchitecture();
    void DisplayLayers();
    void DisplayNeurons();
    void DisplayLayerParametrs(const size_t& kNumLayer);
    void DisplayLayerNeurons(const size_t& kNumLayer);

private:
    std::vector<Layer> m_neuron_layers;
};

