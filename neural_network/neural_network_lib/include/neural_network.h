#pragma once 

#include "pch.h"
#include "layer.h"


class NeuralNetwork
{
public:
    NeuralNetwork() = delete;
    NeuralNetwork(const std::vector<unsigned>& kArchitecture);
    
    Vector<double> CalculateOutputs(const Vector<double> kInputVector);
	Vector<double> FeedForward(const Vector<double> kInputVector);		//Calculate outputs + derivative values
	
    //Setters
    void SetActivationFunction(const unsigned& kNumLayer, const ActivationFunctionType& kActivationFunction);
    void SetSynapticWeigths(const unsigned& kNumLayer, const Matrix<double> kSynapticWeigths);
    void SetBiases(const unsigned& kNumLayer, const Vector<double> kBiases);

    //Getters
    Vector<double> GetOutputs() const;
    Layer GetLayer(const unsigned& kNumLayer) const;
    Layer& GetLayer(const unsigned& kNumLayer);

    size_t GetLayersNumber() const;
    size_t GetInputsNumber() const;
    size_t GetOutputsNumber() const;

    //Displays
    void DisplayArchitecture();
    void DisplayLayers();
    void DisplayNeurons();
    void DisplayLayerParametrs(const unsigned& kNumLayer);
    void DisplayLayerNeurons(const unsigned& kNumLayer);

private:
    std::vector<Layer> m_neuron_layers;
};

