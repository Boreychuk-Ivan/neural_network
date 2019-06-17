#include "neural_network.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& kArchitecture)
{
    size_t layers_number = kArchitecture.size();
	err::assert_throw(layers_number>1, "Error <NeuralNetwork> : Invalid constructor parametrs\n");
    m_neuron_layers.reserve(layers_number);
    for (size_t layer_it = 0; layer_it < layers_number-1; ++layer_it)
    {
		err::assert_throw((kArchitecture.at(layer_it) > 0) && (kArchitecture.at(layer_it + 1) > 0),
			"Error <NeuralNetwork> : Invalid constructor parametrs\n");
        m_neuron_layers.push_back(Layer(kArchitecture.at(layer_it), kArchitecture.at(layer_it + 1), SIGMOID));
    }
    m_neuron_layers.at(layers_number - 2).SetActivationFunction(SIGMOID);  //Output layer
}


Vector<double> NeuralNetwork::CalculateOutputs(const Vector<double> kInputVector)
{
	Vector<double> input_next_layer = m_neuron_layers.at(0).CalculateActivatedValues(kInputVector);
    for (size_t layer_it = 0; layer_it < m_neuron_layers.size()-1; ++layer_it)
    {
		input_next_layer = m_neuron_layers.at(layer_it+1).CalculateActivatedValues(input_next_layer);
    }
    return m_neuron_layers.back().GetActivatedValues();
}

Vector<double> NeuralNetwork::FeedForward(const Vector<double> kInputVector)
{
	Vector<double> input_next_layer = m_neuron_layers.at(0).CalculateActivatedValues(kInputVector);
	auto tmp = m_neuron_layers.at(0).CalculateDerivativeValues(kInputVector);
	for (size_t layer_it = 0; layer_it < m_neuron_layers.size()-1; ++layer_it)
	{
		tmp = m_neuron_layers.at(layer_it + 1).CalculateDerivativeValues(input_next_layer);
		input_next_layer = m_neuron_layers.at(layer_it + 1).CalculateActivatedValues(input_next_layer);
	}
	return m_neuron_layers.back().GetActivatedValues();
}


void NeuralNetwork::SetActivationFunction(const size_t& kNumLayer, const ActivationFunctionType& kActivationFunction)
{
    m_neuron_layers.at(kNumLayer).SetActivationFunction(kActivationFunction);
}

void NeuralNetwork::SetSynapticWeigths(const size_t& kNumLayer, const Matrix<double> kSynapticWeigths)
{
    m_neuron_layers.at(kNumLayer).SetSynapticWeights(kSynapticWeigths);
}

void NeuralNetwork::SetBiases(const size_t& kNumLayer, const Vector<double> kBiases)
{
    m_neuron_layers.at(kNumLayer).SetBiases(kBiases);
}


Vector<double> NeuralNetwork::GetOutputs() const
{
    return m_neuron_layers.back().GetActivatedValues();
}

Layer NeuralNetwork::GetLayer(const size_t & kNumLayer) const
{
    return m_neuron_layers.at(kNumLayer);
}

Layer& NeuralNetwork::GetLayer(const size_t& kNumLayer)
{
    return m_neuron_layers.at(kNumLayer);
}

size_t NeuralNetwork::GetLayersNumber() const
{
    return m_neuron_layers.size();
}

size_t NeuralNetwork::GetInputsNumber() const
{
    return m_neuron_layers.at(0).GetNeuronsNumber();
}

size_t NeuralNetwork::GetOutputsNumber() const
{
    return m_neuron_layers.back().GetNeuronsNumber();
}

void NeuralNetwork::DisplayArchitecture()
{
	std::stringstream out;
	out << "Neural network parametrs:\n";
    out << "Inputs number: "  << m_neuron_layers.at(0).GetNeuronsNumber() << "\n";
    out << "Outputs number: " << m_neuron_layers.back().GetNeuronsNumber() << "\n";
    for(int layer_it = 1; layer_it < m_neuron_layers.size(); ++layer_it)
    {
        out << "Hidden layer #" << layer_it - 1 << " : " 
            << m_neuron_layers.at(layer_it).GetNeuronsNumber() << "\n";
    }
    std::cout << out.str() << "\n";
}

void NeuralNetwork::DisplayLayers()
{
    for (size_t layer_it = 0; layer_it < m_neuron_layers.size(); ++layer_it)
    {
        std::cout << "Layer #" << layer_it << "\n";
        DisplayLayerParametrs(layer_it);
    }
}

void NeuralNetwork::DisplayNeurons()
{
    for (size_t layer_it = 0; layer_it < m_neuron_layers.size(); ++layer_it)
    {
        std::cout << "Layer #" << layer_it << "\n";
        DisplayLayerNeurons(layer_it);
    }
}

void NeuralNetwork::DisplayLayerParametrs(const size_t & kNumLayer)
{
    m_neuron_layers.at(kNumLayer).Display();
}

void NeuralNetwork::DisplayLayerNeurons(const size_t & kNumLayer)
{
    m_neuron_layers.at(kNumLayer).DisplayNeurons();
}
