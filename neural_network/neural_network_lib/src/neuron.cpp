#include "neuron.h"

Neuron::Neuron(double activated_value, ActivationFunctionType type)
    : m_activation_function_type(type), m_activation_function(nullptr), m_derivative_function(nullptr),
      m_local_field(0), m_activated_value(activated_value), m_derivative_value(0),
      m_local_gradient(0)
{
    SetActivationFunction(type);
}

void Neuron::CalculateActivatedValue()
{
    m_activated_value = m_activation_function(m_local_field);
}

void Neuron::CalculateDerivativeValue()
{
    m_derivative_value = m_derivative_function(m_activated_value);
}


void Neuron::Display()
{
	std::stringstream out;
	out << "Neuron parametrs:\n";
	out << "Activation function: " << ActivationFunctions::GetString(m_activation_function_type) << std::endl;
	out << "Local field:" << m_local_field << std::endl;
	out << "Activation value:" << m_activated_value << std::endl;
	out << "Deriative value: " << m_derivative_value << std::endl;
	out << "Local gradient:" << m_local_gradient << std::endl;
	out << std::endl << std::endl;
	std::cout << out.str();
}

void Neuron::SetActivationFunction(ActivationFunctionType type)
{
    if (type == TANH)
    {
        m_activation_function_type = TANH;
        m_activation_function = ActivationFunctions::tanh;
        m_derivative_function = ActivationFunctions::dtanh;
    }
    else if (type == SIGMOID)
    {
        m_activation_function_type = SIGMOID;
        m_activation_function = ActivationFunctions::sigmoid;
        m_derivative_function = ActivationFunctions::dsigmoid;
    }
    else if (type == LINEAR)
    {
        m_activation_function_type = LINEAR;
        m_activation_function = ActivationFunctions::linear;
        m_derivative_function = ActivationFunctions::dlinear;
    }
    else
    {
        throw err::NNException("Error! Invalid activation function\n");
    }
}

void Neuron::SetLocalField(const double& kLocalField)
{
    m_local_field = kLocalField;
}

void Neuron::SetActivatedValue(const double& kActivatedValue)
{
    m_activated_value = kActivatedValue;
}

std::string ActivationFunctions::GetString(ActivationFunctionType type)
{
    if (type == TANH)
    {
        return std::string("TANH");
    }
    else if (type == SIGMOID)
    {
        return std::string("SIGMOID");
    }
    else if (type == LINEAR)
    {
        return std::string("LINEAR");
    }
    else
    {
        return std::string("Warrning! Invalid activation function\n");
    }
}

ActivationFunctionType ActivationFunctions::Type(const std::string& kType)
{
    if (kType == "TANH")
    {
        return TANH;
    }
    else if (kType == "SIGMOID")
    {
        return SIGMOID;
    }
    else if (kType == "LINEAR")
    {
        return LINEAR;
    }
    else
    {
		throw err::NNException("Error! Invalid activation function\n");
    }
}
