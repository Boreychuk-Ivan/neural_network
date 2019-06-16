#pragma once
#include "pch.h"
#include "activation_functions.h"


class Neuron
{
public:
    Neuron() = delete;
    Neuron(double activated_value, ActivationFunctionType type);
    
	double CalculateActivatedValue(const double& kLocalField);
	double CalculateDerivativeValue(const double& kLocalField);

    void SetActivationFunction(ActivationFunctionType);
    void SetLocalField(const double&);
    void SetActivatedValue(const double&);

    ActivationFunctionType GetActivationFunctionType() const { return m_activation_function_type; }
	double GetLocalFiled() const { return m_local_field; };
	double GetActivatedValue() const { return m_activated_value; }
	double GetDerivativeValue() const { return m_derivative_value; }
    
    void Display();

private:
    ActivationFunctionType m_activation_function_type;
    ActivationFunction m_activation_function;
    DerivativeFunction m_derivative_function;
    double m_local_field;
    double m_activated_value;
    double m_derivative_value;

};

