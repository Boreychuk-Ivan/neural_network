#pragma once
#include "pch.h"
#include "neuron.h"


class Layer
{
public:
    Layer() = delete;
    Layer(const unsigned& inputs_number, const unsigned& neurons_number,
          const ActivationFunctionType& kActivationFunction);

    void InitializeRandomWeights(const double& min_value, const double& max_value);
    void InitializeRandomBiases(const double& min_value, const double& max_value);
    
    void AdjustmentWeights(const Matrix<double> kDeltaWeights);
    void AdjustmentBiases(const Vector<double> kDeltaBiases);
    Vector<double> CalculateLocalFields(const Vector<double> kInputVector);
    Vector<double> CalculateActivatedValues(const Vector<double> kInputVector);
    Vector<double> CalculateDerivativeValues(const Vector<double> kInputVector);

    void SetActivationFunction(const ActivationFunctionType& kActivationFunction);
    void SetLocalField(const Vector<double> kLocalFieldVector);
    void SetActivatedValues(const Vector<double> kActivatedValueVector);
    void SetSynapticWeights(const Matrix<double> kSynapticWeigths);
    void SetBiases(const Vector<double> kBiases);

    size_t GetNeuronsNumber() const;
    size_t GetInputsNumber() const;
    Vector<double> GetLocalFields() const;
    Vector<double> GetActivatedValues() const;
    Vector<double> GetDerivativeValues() const;
    Matrix<double> GetSynapticWeights() const;
    Vector<double> GetBiases() const;
    ActivationFunctionType GetActivationFunctionType() const;

    void Display();
    void DisplayNeurons();

protected:
    std::vector<Neuron> m_neurons;
private:
    Matrix<double> m_synaptic_weights;
    Vector<double> m_biases;
};