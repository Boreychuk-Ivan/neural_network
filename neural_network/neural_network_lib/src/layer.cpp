#include "layer.h"

Layer::Layer(const unsigned& inputs_number, const unsigned& neurons_number, const ActivationFunctionType& activation_function) :
    m_neurons(neurons_number, Neuron(0, activation_function)), m_synaptic_weights(neurons_number, inputs_number),
    m_delta_weights(neurons_number, inputs_number), m_biases(neurons_number)
{
    InitializeRandomWeights(-5,5);
    InitializeRandomBiases(-5, 5);
}

void Layer::InitializeRandomWeights(const double& min_value, const double& max_value)
{
    assert(max_value > min_value);
    int precision = 100;
    for(int it = 0; it < m_synaptic_weights.GetSize(); ++it)
    {
        m_synaptic_weights.at(it) = min_value + fmod(std::rand(), (max_value - min_value) * precision) / precision;
    }
}

void Layer::InitializeRandomBiases(const double& min_value, const double& max_value)
{
    assert(max_value > min_value);
    int precision = 100;
    for (int it = 0; it < m_biases.GetSize(); ++it)
    {
        m_biases.at(it) = min_value + fmod(std::rand(), (max_value - min_value)* precision)/ precision;
    }
}

void Layer::CalculateLocalFields(const Vector<double>& kInputVector)
{
    assert(kInputVector.IsCol() == 1);
    unsigned neurons_number = m_neurons.size();
    Vector<double> local_field(neurons_number);
    local_field = (m_synaptic_weights * kInputVector + m_biases);
    SetLocalField(local_field);
}

void Layer::CalculateActivatedValues()
{
    for(auto& neuron : m_neurons)
        neuron.CalculateActivatedValue();
}

void Layer::CalculateDerivativeValues()
{
    for (auto& neuron : m_neurons)
        neuron.CalculateDerivativeValue();
}

void Layer::SetLocalField(const Vector<double> kLocalFieldVector)
{
    int neurons_num = m_neurons.size();
    assert(neurons_num == kLocalFieldVector.GetSize());
    for (int it = 0; it < neurons_num; ++it)
    {
        m_neurons.at(it).SetLocalField(kLocalFieldVector.at(it));
    }
}

void Layer::SetActivatedValues(const Vector<double> kActivatedValueVector)
{
    assert(m_neurons.size() == kActivatedValueVector.GetSize());
    for(int it = 0; it < m_neurons.size(); ++it)
    {
        m_neurons.at(it).SetActivatedValue(kActivatedValueVector.at(it));
    }
}

void Layer::SetBiases(const Vector<double> kBiases)
{
    assert(m_neurons.size() == kBiases.GetSize());
    for (int it = 0; it < m_neurons.size(); ++it)
    {
        m_biases.at(kBiases.at(it));
    }
}

void Layer::Display()
{
    std::cout << "Layer parametrs:\n";
    std::cout << "Neurons number: " << m_neurons.size() << "\n";
    std::cout << "Activation function:" << 
        Functions::Disp(m_neurons.at(1).GetActivationFunctionType()) << "\n";
    std::cout << "Number of inputs: " << m_synaptic_weights.GetNumCols() << "\n";
    std::cout << "Biases : " << m_biases;
    std::cout << "Synaptic weights" << m_synaptic_weights << "\n";
    std::cout << "Delta weights" << m_delta_weights << "\n";
}

void Layer::DisplayNeurons()
{
    std::cout << "Layer neurons parametrs:\n";
    std::cout << "lf\tav\tdv\n";
    for (int it = 0; it < m_neurons.size(); ++it)
    {
        std::cout 
            << (m_neurons.at(it).GetLocalFiled() > 0 ? "" : " ")
            << m_neurons.at(it).GetLocalFiled() << "\t"
            << (m_neurons.at(it).GetActivatedValue() > 0 ? "" : " ")
            << m_neurons.at(it).GetActivatedValue() << "\t"
            << (m_neurons.at(it).GetDerivativeValue() > 0 ? "" : " ")
            << m_neurons.at(it).GetDerivativeValue() << "\n";
    }
}
