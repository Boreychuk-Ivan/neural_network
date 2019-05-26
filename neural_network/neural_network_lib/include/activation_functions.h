#pragma once
#include <cmath>

enum ActivationFunctionType
{
    TANH,
    SIGMOID,
    LINEAR,
};

using ActivationFunction = double(*)(double);
using DerivativeFunction = double(*)(double);

class ActivationFunctions
{
public:
    static double tanh(double x) { return std::tanh(x); }
    static double sigmoid(double x) { return (1 / (1 + exp(-x))); }
    static double linear(double x) { return x; }

    static double dtanh(double x) { return (1 - x * x); }
    static double dsigmoid(double x) { return (x * (1 - x)); }
    static double dlinear(double x) { return 1; }

    static std::string GetString(ActivationFunctionType);
    static ActivationFunctionType Type(const std::string& kType);
};