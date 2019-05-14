#include "input_layer.h"

InputLayer::InputLayer(const unsigned& neurons_number, const ActivationFunctionType& activation_function):
    Layer(0, neurons_number, activation_function)
{
}

void InputLayer::PushInputData(const Vector<double>& kInputVector)
{
    SetLocalField(kInputVector);
    CalculateActivatedValues();
}

void InputLayer::Display()
{
    std::cout << "Input layer parametrs:\n";
    std::cout << "Neurons number: " << m_neurons.size() << "\n";
    std::cout << "Activation function:" <<
        Functions::Disp(m_neurons.at(1).GetActivationFunctionType()) << "\n";
}
