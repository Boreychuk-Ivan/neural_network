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
    
    void AdjustmentWeight(const Vector<double> kDeltaWeights);
    Vector<double> CalculateLocalFields(Vector<double> input_vector);
    Vector<double> CalculateActivatedValues();
    Vector<double> CalculateDerivativeValues();

    void SetActivationFunction(const ActivationFunctionType& kActivationFunction);
    void SetLocalField(const Vector<double> kLocalFieldVector);
    void SetActivatedValues(const Vector<double> kActivatedValueVector);
    void SetSynapticWeights(const Matrix<double> kSynapticWeigths);
    void SetDeltaWeigths(const Matrix<double>& kDeltaWeights);
    void SetBiases(const Vector<double> kBiases);

    size_t GetNeuronsNumber() const;
    size_t GetInputsNumber() const;
    Vector<double> GetLocalField() const;
    Vector<double> GetActivatedValues() const;
    Vector<double> GetDerivativeValues() const;
    Matrix<double> GetSynapticWeights() const;
    Matrix<double> GetDeltaWeigths() const;
    Vector<double> GetBiases() const;

    void Display();
    void DisplayNeurons();

};