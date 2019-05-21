#pragma once
#include "pch.h"
#include "neuron.h"


class Layer
{
protected:
    std::vector<Neuron> m_neurons;
private:
    Matrix<double> m_synaptic_weights;
    Matrix<double> m_delta_weights;
    Vector<double> m_biases;
public:
    Layer() = delete;
    Layer(const unsigned& inputs_number, const unsigned& neurons_number,
          const ActivationFunctionType& kActivationFunction);

    void InitializeRandomWeights(const double& min_value, const double& max_value);
    void InitializeRandomBiases(const double& min_value, const double& max_value);

    void CalculateLocalFields(Vector<double> input_vector);
    void CalculateActivatedValues();
    void CalculateDerivativeValues();

    void SetActivationFunction(const ActivationFunctionType& kActivationFunction);
    void SetLocalField(const Vector<double> kLocalFieldVector);
    void SetActivatedValues(const Vector<double> kActivatedValueVector);
    void SetSynapticWeights(const Matrix<double> kSynapticWeigths);
    void SetBiases(const Vector<double> kBiases);

    unsigned GetNeuronsNumber();
    Vector<double> GetLocalField();
    Vector<double> GetActivatedValues();
    Vector<double> GetDerivativeValues();

    void Display();
    void DisplayNeurons();

};