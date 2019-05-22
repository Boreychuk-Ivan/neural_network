#include "layer.h"

Layer::Layer(const unsigned& inputs_number, const unsigned& neurons_number, const ActivationFunctionType& activation_function) :
    m_neurons(neurons_number, Neuron(0, activation_function)), m_synaptic_weights(neurons_number, inputs_number),
    m_delta_weights(neurons_number, inputs_number), m_biases(neurons_number)
{
    InitializeRandomWeights(-2,2);      //Liniar part of sigmoid function
    InitializeRandomBiases(-2, 2);
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

void Layer::AdjustmentWeight(const Vector<double> kDeltaWeights)
{
    SetDeltaWeigths(kDeltaWeights);
    m_synaptic_weights = m_synaptic_weights + kDeltaWeights;
}

Vector<double> Layer::CalculateLocalFields(Vector<double> input_vector)
{
    unsigned inputs_number = m_synaptic_weights.GetColsNum();
    assert(input_vector.GetSize() == inputs_number);
    if (input_vector.IsRow())
        input_vector = !input_vector; //Transpose
    unsigned neurons_number = m_neurons.size();
    Vector<double> local_field(neurons_number);
    local_field = (m_synaptic_weights * input_vector + !m_biases);
    return local_field;
}

Vector<double> Layer::CalculateActivatedValues()
{
    for(auto& neuron : m_neurons)
        neuron.CalculateActivatedValue();
    return GetActivatedValues();
}

Vector<double> Layer::CalculateDerivativeValues()
{
    for (auto& neuron : m_neurons)
        neuron.CalculateDerivativeValue();
    return GetDerivativeValues();
}

void Layer::SetActivationFunction(const ActivationFunctionType & kActivationFunction)
{
    for (auto& neuron : m_neurons)
        neuron.SetActivationFunction(kActivationFunction);
}

void Layer::SetLocalField(const Vector<double> kLocalFieldVector)
{
    int neurons_num = m_neurons.size();
    assert(neurons_num == kLocalFieldVector.GetSize());
    for (int it = 0; it < neurons_num; ++it)
        m_neurons.at(it).SetLocalField(kLocalFieldVector.at(it));
}

void Layer::SetActivatedValues(const Vector<double> kActivatedValueVector)
{
    assert(m_neurons.size() == kActivatedValueVector.GetSize());
    for(int it = 0; it < m_neurons.size(); ++it)
        m_neurons.at(it).SetActivatedValue(kActivatedValueVector.at(it));
}

void Layer::SetSynapticWeights(const Matrix<double> kSynapticWeigths)
{
    assert(m_synaptic_weights.IsEqualSize(kSynapticWeigths));
    m_synaptic_weights = kSynapticWeigths;
}

void Layer::SetDeltaWeigths(const Matrix<double>& kDeltaWeights)
{
    assert(kDeltaWeights.GetColsNum() == m_delta_weights.GetColsNum());
    assert(kDeltaWeights.GetRowsNum() == m_delta_weights.GetRowsNum());
    m_delta_weights = kDeltaWeights;
}


void Layer::SetBiases(const Vector<double> kBiases)
{
    assert(m_neurons.size() == kBiases.GetSize());
    for (int it = 0; it < m_neurons.size(); ++it)
        m_biases.at(it) = kBiases.at(it);
}

size_t Layer::GetNeuronsNumber() const
{
    return m_neurons.size();
}

size_t Layer::GetInputsNumber() const
{
    return m_delta_weights.GetColsNum();
}

Vector<double> Layer::GetLocalField() const
{
    Vector<double> local_fields(m_neurons.size());
    for (int neuron_it = 0; neuron_it < m_neurons.size(); ++neuron_it)
        local_fields.at(neuron_it) = m_neurons.at(neuron_it).GetLocalFiled();
    return local_fields;
}

Vector<double> Layer::GetActivatedValues() const
{
    Vector<double> activated_values(m_neurons.size());
    for(int neuron_it = 0; neuron_it < m_neurons.size(); ++neuron_it)
        activated_values.at(neuron_it) = m_neurons.at(neuron_it).GetActivatedValue();
    return activated_values;
}

Vector<double> Layer::GetDerivativeValues() const
{
    Vector<double> derivative_values(m_neurons.size());
    for (int neuron_it = 0; neuron_it < m_neurons.size(); ++neuron_it)
        derivative_values.at(neuron_it) = m_neurons.at(neuron_it).GetDerivativeValue();
    return derivative_values;
}

Matrix<double> Layer::GetSynapticWeights() const
{
    return m_synaptic_weights;
}

Matrix<double> Layer::GetDeltaWeigths() const
{
    return m_delta_weights;
}

Vector<double> Layer::GetBiases() const
{
    return m_biases;
}

void Layer::Display()
{
    std::cout << "Layer parametrs:\n";
    std::cout << "Neurons number: " << m_neurons.size() << "\n";
    std::cout << "Activation function:" << 
        Functions::Disp(m_neurons.at(1).GetActivationFunctionType()) << "\n";
    std::cout << "Number of inputs: " << m_synaptic_weights.GetColsNum() << "\n";
    std::cout << "Biases : " << m_biases << "\n";
    std::cout << "Synaptic weights" << m_synaptic_weights << "\n";
    std::cout << "Delta weights" << m_delta_weights << "\n";
    std::cout << "\n";
}

void Layer::DisplayNeurons()
{
    std::cout << "Layer neurons parametrs:\n";
    std::cout << "lf\tav\tdv\n";
    std::cout.precision(3);
    std::fixed;
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
    std::cout << "\n";
}
