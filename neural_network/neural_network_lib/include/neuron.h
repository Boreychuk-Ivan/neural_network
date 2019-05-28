#pragma once
#include "pch.h"
#include "activation_functions.h"


class Neuron
{
public:
    Neuron() = delete;
    Neuron(double activated_value, ActivationFunctionType type);
    
    void CalculateActivatedValue();
    void CalculateDerivativeValue();

    void SetActivationFunction(ActivationFunctionType);
    void SetLocalField(const double&);
    void SetActivatedValue(const double&);

    auto GetActivationFunctionType() const { return m_activation_function_type; }
    auto GetLocalFiled() const { return m_local_field; };
    auto GetActivatedValue() const { return m_activated_value; }
    auto GetDerivativeValue() const { return m_derivative_value; }
    auto GetLocalGradient()const { return m_local_gradient;  }
    
    void Display();

private:
    ActivationFunctionType m_activation_function_type;
    ActivationFunction m_activation_function;
    DerivativeFunction m_derivative_function;
    double m_local_field;
    double m_activated_value;
    double m_derivative_value;
    double m_local_gradient;

};

