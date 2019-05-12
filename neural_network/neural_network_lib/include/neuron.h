#pragma once
#include "pch.h"
#include <cmath>

enum ActivationFunctionType
{
    TANH,
    SIGMOID,
    IDENTITY,
};

class Functions;
using ActivationFunction = double(*)(double);
using DerivativeFunction = double(*)(double);

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

class Functions
{
public:
    static double tanh(double x) { return std::tanh(x); }
    static double sigmoid(double x) { return (1 / (1 + exp(-x))); }
    static double identity(double x) { return x; }

    static double dtanh(double x) { return (1 - x * x); }
    static double dsigmoid(double x) { return (x * (1 - x)); }
    static double didentity(double x) { return 1; }

    static std::string Disp(ActivationFunctionType);
};