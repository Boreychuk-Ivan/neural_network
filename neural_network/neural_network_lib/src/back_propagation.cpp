#include "back_propagation.h"

BackPropagation::BackPropagation(NeuralNetwork& kNeuralNetwork, const double kLearningRate, const double kMomentum) : 
    m_neural_network(kNeuralNetwork), 
    m_learning_rate(kLearningRate), 
    m_momentum(kMomentum), 
    m_delta_weights(kNeuralNetwork.GetLayersNumber(),Matrix<double>()),
    m_delta_biases(kNeuralNetwork.GetLayersNumber(),Vector<double>())
{

    for (size_t it = 0; it < kNeuralNetwork.GetLayersNumber(); ++it)
    {
        Layer& cur_layer = kNeuralNetwork.GetLayer(it);
        m_delta_weights.at(it) = Matrix<double>(cur_layer.GetNeuronsNumber(), cur_layer.GetInputsNumber());
        m_delta_biases.at(it) = Vector<double>(cur_layer.GetNeuronsNumber());
    }
}

Vector<double> BackPropagation::CalculateLocalGradients(
    const Vector<double>& kDerivativeValues, 
    const Vector<double>& kPreviousLayerError, 
    const Matrix<double>& kSynapticWeights)
{
    Matrix<double> tmp = kPreviousLayerError * kSynapticWeights;
    return Vector<double>(kDerivativeValues.DotMult(tmp));
}

Matrix<double> BackPropagation::CalculateDeltaWeights(
    const double& kLearningRate, 
    const double& kMomentum, 
    const Vector<double>& kLocalGradients, 
    const Vector<double>& kInputValues, 
    const Matrix<double>& kLastDeltaWeigths)
{
    return (kMomentum * kLastDeltaWeigths + kLearningRate * !kLocalGradients * kInputValues); 
}

Vector<double> BackPropagation::CalculateDeltaBiases(const double& kLearningRate, const double& kMomentum, const Vector<double>& kLocalGradients, const Vector<double>& kLastDeltaBiases)
{
    return (kMomentum* kLastDeltaBiases + kLearningRate* kLocalGradients);
}

void BackPropagation::AdjustmentNeuralNetwork(const Vector<double>& kInputData, const Vector<double>& kTargetValues)
{
    size_t layers_number = m_neural_network.GetLayersNumber();
    
    //Feed forward
    m_neural_network.CalculateOutputs(kInputData);
    Vector<double> error = kTargetValues - m_neural_network.GetOutputs();

    //Back propagation
    Matrix<double> weights;
    Vector<double> out_error;
    Vector<double> derivative_values;
    Vector<double> local_gradients;
    Vector<double> input_vector;
    
    Matrix<double> delta_weights;
    Matrix<double> last_delta_weights;
    Vector<double> delta_biases;
    Vector<double> last_delata_biases;
    
    //Out layer
    derivative_values = m_neural_network.GetLayer(layers_number-1).CalculateDerivativeValues();
    local_gradients = error.DotMult(derivative_values);    

    //Hidden layers
    for (int it = layers_number-1; it > 0; --it)    
    {
        //Adjustment weights
        derivative_values = m_neural_network.GetLayer(it - 1).CalculateDerivativeValues();
        weights = m_neural_network.GetLayer(it).GetSynapticWeights();
        input_vector = m_neural_network.GetLayer(it-1).GetActivatedValues();
        last_delta_weights = m_delta_weights.at(it);
        delta_weights = CalculateDeltaWeights(m_learning_rate, m_momentum, local_gradients, input_vector, last_delta_weights);
        m_delta_weights.at(it) = delta_weights;
        m_neural_network.GetLayer(it).AdjustmentWeights(delta_weights);
        //Adjustment biases
        last_delata_biases = m_delta_biases.at(it);
        delta_biases = CalculateDeltaBiases(m_learning_rate, m_momentum, local_gradients, last_delata_biases);
        m_delta_biases.at(it) = delta_biases;
        m_neural_network.GetLayer(it).AdjustmentBiases(delta_biases);
        local_gradients = CalculateLocalGradients(derivative_values, local_gradients, weights);
    }

    //Input layer
    weights = m_neural_network.GetLayer(0).GetSynapticWeights();
    last_delta_weights = m_delta_weights.at(0);
    delta_weights = CalculateDeltaWeights(m_learning_rate, m_momentum, local_gradients, kInputData, last_delta_weights);
    m_neural_network.GetLayer(0).AdjustmentWeights(delta_weights);
    m_delta_weights.at(0) = delta_weights;

    last_delata_biases = m_neural_network.GetLayer(0).GetBiases();
    delta_biases = CalculateDeltaBiases(m_learning_rate, m_momentum, local_gradients, last_delata_biases);
    m_neural_network.GetLayer(0).AdjustmentBiases(delta_biases);
}


NeuralNetwork BackPropagation::GetNeuralNetwork() const
{
    return m_neural_network;
}

Matrix<double> BackPropagation::GetDeltaWeights(const size_t kLayerNum) const
{
    return m_delta_weights.at(kLayerNum);
}

Vector<double> BackPropagation::GetDeltaBiases(const size_t kLayerNum) const
{
    return m_delta_biases.at(kLayerNum);
}
