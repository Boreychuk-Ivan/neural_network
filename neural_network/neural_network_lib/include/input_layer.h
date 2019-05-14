#pragma once
#include "pch.h"
#include "layer.h"

class InputLayer : public Layer
{
private:
    using Layer::m_neurons;
public:
    //unused layer class methods
    InputLayer() = delete;
    void InitializeRandomWeights(const double&, const double&) = delete;
    void InitializeRandomBiases(const double&, const double&) = delete;
    void SetBiases(const Vector<double>) = delete;
    void CalculateLocalFields(const Vector<double>&) = delete;

    //used layer class methods
    using Layer::CalculateActivatedValues;
    using Layer::CalculateDerivativeValues;
    using Layer::SetActivatedValues;
    using Layer::SetLocalField;
    using Layer::DisplayNeurons;

    //new methods
    InputLayer(const unsigned& neurons_number, const ActivationFunctionType& activation_function);
    void PushInputData(const Vector<double>& kInputVector);

    void Display();
};