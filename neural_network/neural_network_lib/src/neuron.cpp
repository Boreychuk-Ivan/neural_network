#include "neuron.h"

Neuron::Neuron(double activated_value, ActivationFunctionType type)
    : m_activation_function_type(type), m_activation_function(nullptr), m_derivative_function(nullptr),
      m_local_field(0), m_activated_value(activated_value), m_deriative_value(0),
      m_local_gradient(0)
{
    SetActivationFunction(type);
}

void Neuron::CalculateActivatedValues()
{
    m_activated_value = m_activation_function(m_local_field);
}

void Neuron::CalculateDerivativeValue()
{
    m_deriative_value = m_derivative_function(m_activated_value);
}


void Neuron::Display()
{
    std::cout << "Neuron parametrs:\n";
    std::cout << "Activation function: " << Functions::Disp(m_activation_function_type) << std::endl;
    std::cout << "Local field:" << m_local_field << std::endl;
    std::cout << "Activation value:" << m_activated_value << std::endl;
    std::cout << "Deriative value: " << m_deriative_value << std::endl;
    std::cout << "Local gradient:" << m_local_gradient << std::endl;
    std::cout << std::endl << std::endl;
}

void Neuron::SetActivationFunction(ActivationFunctionType type)
{
    if (type == TANH)
    {
        m_activation_function = Functions::tanh;
        m_derivative_function = Functions::dtanh;
    }
    else if (type == SIGMOID)
    {
        m_activation_function = Functions::sigmoid;
        m_derivative_function = Functions::dsigmoid;
    }
    else if (type == LINEAR)
    {
        m_activation_function = Functions::linear;
        m_derivative_function = Functions::dlinear;
    }
    else
    {
        std::cerr << "Error! Invalid activation function\n";
        exit(1);
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

std::string Functions::Disp(ActivationFunctionType type)
{
    if (type == TANH)
    {
        return std::string("TANH");
    }
    if (type == SIGMOID)
    {
        return std::string("SIGMOID");
    }
    if (type == LINEAR)
    {
        return std::string("LINEAR");
    }
    else
    {
        return std::string("Warrning! Invalid activation function\n");
    }
    
}
