#pragma once
#include "pch.h"
#include "neuron.h"
#include "matrix_lib/matrix_lib.h"

class Layer
{
private:
    std::vector<Neuron> m_neurons;
    Matrix<double> m_synaptic_weights;
    Matrix<double> m_delta_weights; //??
    Vector<double, VTYPE::COL> m_biases;
public:
    Layer() = delete;
    Layer(const unsigned& inputs_number, const unsigned& neurons_number,
          const ActivationFunctionType& activation_function);

    void InitializeRandomWeights(const double& min_value, const double& max_value);
    void InitializeRandomBiases(const double& min_value, const double& max_value);

    void CalculateLocalFields(const Vector<double>& kInputVector);
    void CalculateActivatedValues();
    void CalculateDerivativeValue();

    void SetLocalField(const Vector<double> kLocalFieldVector);
    void SetActivatedValues(const Vector<double> kActivatedValueVector);
    void SetBiases(const Vector<double> kBiases);

    void Display();
};