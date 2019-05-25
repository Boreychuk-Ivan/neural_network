#include "neural_network.h"

NeuralNetwork::NeuralNetwork(const std::vector<unsigned>& kArchitecture)
{
    unsigned layers_number = kArchitecture.size();   
    m_neuron_layers.reserve(layers_number);
    for (size_t layer_it = 0; layer_it < layers_number-1; ++layer_it)
    {
        m_neuron_layers.push_back(Layer(kArchitecture.at(layer_it), kArchitecture.at(layer_it + 1), SIGMOID));
    }
    m_neuron_layers.at(layers_number - 2).SetActivationFunction(SIGMOID);  //Output layer
}


Vector<double> NeuralNetwork::CalculateOutputs(const Vector<double> kInputVector)
{
    m_neuron_layers.at(0).CalculateLocalFields(kInputVector);
    for (size_t layer_it = 0; layer_it < m_neuron_layers.size()-1; ++layer_it)
    {
        Vector<double> input_next_layer = m_neuron_layers.at(layer_it).CalculateActivatedValues();
        m_neuron_layers.at(layer_it+1).CalculateLocalFields(input_next_layer);
    }
    return m_neuron_layers.back().CalculateActivatedValues();
}

void NeuralNetwork::CalculateDerivativeValues()
{
    for (auto& layer : m_neuron_layers)
        layer.CalculateDerivativeValues();
}

Vector<double> NeuralNetwork::CalculateDerivativeValuesOnLayer(const unsigned& kNumLayer)
{
    return m_neuron_layers.at(kNumLayer).CalculateDerivativeValues();
}

void NeuralNetwork::SetActivationFunction(const unsigned& kNumLayer, const ActivationFunctionType& kActivationFunction)
{
    m_neuron_layers.at(kNumLayer).SetActivationFunction(kActivationFunction);
}

void NeuralNetwork::SetSynapticWeigths(const unsigned& kNumLayer, const Matrix<double> kSynapticWeigths)
{
    m_neuron_layers.at(kNumLayer).SetSynapticWeights(kSynapticWeigths);
}

void NeuralNetwork::SetBiases(const unsigned& kNumLayer, const Vector<double> kBiases)
{
    m_neuron_layers.at(kNumLayer).SetBiases(kBiases);
}


Vector<double> NeuralNetwork::GetOutputs() const
{
    return m_neuron_layers.back().GetActivatedValues();
}

Layer NeuralNetwork::GetLayer(const unsigned & kNumLayer) const
{
    return m_neuron_layers.at(kNumLayer);
}

Layer& NeuralNetwork::GetLayer(const unsigned& kNumLayer)
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
    std::cout << "Neural network parametrs:\n";
    std::cout << "Inputs number: "  << m_neuron_layers.at(0).GetNeuronsNumber() << "\n";
    std::cout << "Outputs number: " << m_neuron_layers.back().GetNeuronsNumber() << "\n";
    for(int layer_it = 1; layer_it < m_neuron_layers.size(); ++layer_it)
    {
        std::cout << "Hidden layer #" << layer_it - 1 << " : " 
            << m_neuron_layers.at(layer_it).GetNeuronsNumber() << "\n";
    }
    std::cout << "\n";
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

void NeuralNetwork::DisplayLayerParametrs(const unsigned & kNumLayer)
{
    m_neuron_layers.at(kNumLayer).Display();
}

void NeuralNetwork::DisplayLayerNeurons(const unsigned & kNumLayer)
{
    m_neuron_layers.at(kNumLayer).DisplayNeurons();
}
