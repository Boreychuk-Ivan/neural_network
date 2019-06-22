#include "layer.h"

Layer::Layer(const int& kInputsNumber, const int& kNeuronsNumber, const ActivationFunctionType& activation_function) :
    m_neurons(kNeuronsNumber, Neuron(0, activation_function)), m_synaptic_weights(kNeuronsNumber, kInputsNumber),
    m_biases(kNeuronsNumber)
{
	err::assert_throw((kInputsNumber > 0) && (kNeuronsNumber > 0), "Error <Layer> : Invalid constructor parametrs\n");
    InitializeRandomWeights(-1,1);      //Liniar part of sigmoid function
    InitializeRandomBiases(-1, 1);
}

void Layer::InitializeRandomWeights(const double& min_value, const double& max_value)
{
    err::assert_throw(max_value > min_value,"Error <InitializeRandomWeights>: max_value < min_value\n");
    int precision = 100;
    for(int it = 0; it < m_synaptic_weights.GetSize(); ++it)
    {
        m_synaptic_weights.at(it) = min_value + fmod(std::rand(), (max_value - min_value) * precision) / precision;
    }
}

void Layer::InitializeRandomBiases(const double& min_value, const double& max_value)
{
	err::assert_throw(max_value > min_value, "Error <InitializeRandomBiases>: max_value < min_value\n");
    int precision = 100;
    for (int it = 0; it < m_biases.GetSize(); ++it)
    {
        m_biases.at(it) = min_value + fmod(std::rand(), (max_value - min_value)* precision)/ precision;
    }
}

void Layer::AdjustmentWeights(const Matrix<double> kDeltaWeights)
{
    m_synaptic_weights = m_synaptic_weights + kDeltaWeights;
}

void Layer::AdjustmentBiases(const Vector<double> kDeltaBiases)
{
    m_biases = m_biases + kDeltaBiases;
}

Vector<double> Layer::CalculateLocalFields(const Vector<double> kInputVector)
{
    unsigned inputs_number = m_synaptic_weights.GetColsNum();
	err::assert_throw(kInputVector.GetSize() == inputs_number, "Error <CalculateLocalFields> : Invalid input vector\n");

	Vector<double> inputs = (kInputVector.IsRow()) ? !kInputVector : kInputVector;

    unsigned neurons_number = m_neurons.size();
    Vector<double> local_field(neurons_number);
    local_field = (m_synaptic_weights * inputs + !m_biases);
    SetLocalField(local_field);
    return local_field;
}

Vector<double> Layer::CalculateActivatedValues(const Vector<double> kInputVector)
{
	auto local_field = CalculateLocalFields(kInputVector);
	for (size_t it = 0; it < m_neurons.size(); ++it)
		m_neurons.at(it).CalculateActivatedValue(local_field.at(it));
    return GetActivatedValues();
}

Vector<double> Layer::CalculateDerivativeValues(const Vector<double> kInputVector)
{
	auto local_field = CalculateLocalFields(kInputVector);
	for (size_t it = 0; it < m_neurons.size(); ++it)
		m_neurons.at(it).CalculateDerivativeValue(local_field.at(it));
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
	err::assert_throw(neurons_num == kLocalFieldVector.GetSize(), "Error <SetLocalField> : Invalid input vector\n");
    for (int it = 0; it < neurons_num; ++it)
        m_neurons.at(it).SetLocalField(kLocalFieldVector.at(it));
}

void Layer::SetActivatedValues(const Vector<double> kActivatedValueVector)
{
	err::assert_throw(m_neurons.size() == kActivatedValueVector.GetSize(),"Error <SetActivatedValues> : Invalid input vector\n");
    for(int it = 0; it < m_neurons.size(); ++it)
        m_neurons.at(it).SetActivatedValue(kActivatedValueVector.at(it));
}

void Layer::SetSynapticWeights(const Matrix<double> kSynapticWeigths)
{
	err::assert_throw(m_synaptic_weights.IsEqualSize(kSynapticWeigths), "Error <SetSynapticWeights> : Invalid input matrix\n");
    m_synaptic_weights = kSynapticWeigths;
}



void Layer::SetBiases(const Vector<double> kBiases)
{
	err::assert_throw(m_neurons.size() == kBiases.GetSize(), "Error <SetBiases> : Invalid input vector\n");
    for (int it = 0; it < m_neurons.size(); ++it)
        m_biases.at(it) = kBiases.at(it);
}


size_t Layer::GetNeuronsNumber() const
{
    return m_neurons.size();
}

size_t Layer::GetInputsNumber() const
{
    return m_synaptic_weights.GetColsNum();
}

Vector<double> Layer::GetLocalFields() const
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


Vector<double> Layer::GetBiases() const
{
    return m_biases;
}


ActivationFunctionType Layer::GetActivationFunctionType() const
{
    return m_neurons.at(0).GetActivationFunctionType();
}

void Layer::Display()
{
	std::stringstream out;
	out << "Layer parametrs:\n";
	out << "Neurons number: " << m_neurons.size() << "\n";
	out << "Number of inputs: " << m_synaptic_weights.GetColsNum() << "\n";
	out << "Activation function:";
    out << ActivationFunctions::GetString(m_neurons.at(0).GetActivationFunctionType()) << "\n";
	out << "Biases : " << m_biases << "\n";
	out << "Synaptic weights" << m_synaptic_weights << "\n\n";
    std::cout << out.str();
}

void Layer::DisplayNeurons()
{
	std::stringstream out;
	out << "Layer neurons parametrs:\n";
	out << "lf\tav\tdv\n";
	out.precision(5);
	out << std::fixed;
    for (int it = 0; it < m_neurons.size(); ++it)
    {
		out
            << (m_neurons.at(it).GetLocalFiled() > 0 ? "" : " ")
            << m_neurons.at(it).GetLocalFiled() << "\t"
            << (m_neurons.at(it).GetActivatedValue() > 0 ? "" : " ")
            << m_neurons.at(it).GetActivatedValue() << "\t"
            << (m_neurons.at(it).GetDerivativeValue() > 0 ? "" : " ")
            << m_neurons.at(it).GetDerivativeValue() << "\n";
    }
    std::cout << out.str() << "\n";
}
