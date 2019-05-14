#pragma once
#include "pch.h"
#include "activation_functions.h"


class Neuron
{
private:
    ActivationFunctionType m_activation_function_type;
    ActivationFunction m_activation_function;
    DerivativeFunction m_derivative_function;
    double m_local_field;
    double m_activated_value;
    double m_deriative_value;
    double m_local_gradient;

public:
    Neuron() = delete;
    Neuron(double activated_value, ActivationFunctionType type);
    
    void CalculateActivatedValues();
    void CalculateDerivativeValue();
    void Display();

    void SetActivationFunction(ActivationFunctionType);
    void SetLocalField(const double&);
    void SetActivatedValue(const double&);
};

