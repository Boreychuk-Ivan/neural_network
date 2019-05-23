#include "back_propagation.h"

Vector<double> BackPropagation::CalculateError(
    const Vector<double>& kOutputValues, 
    const Vector<double>& kTargetValues)
{
    assert(kOutputValues.IsEqualSize(kTargetValues));
    m_error = kTargetValues - kOutputValues;
    return m_error;
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
    //Feed forward
    m_neural_network.PushInputData(kInputData);
    m_neural_network.CalculateOutputs();

    size_t layers_number = m_neural_network.GetLayersNumber();
    Matrix<double> weights;
    weights.InitialiseDiag(m_neural_network.GetOutputsNumber());
    Vector<double> out_error;
    Vector<double> derivative_values;
    Vector<double> local_gradients;
    Vector<double> input_vector;
    
    Matrix<double> delta_weights;
    Matrix<double> last_delta_weights;
    Vector<double> delta_biases;
    Vector<double> last_delata_biases;
    
    Vector<double> error = CalculateError(m_neural_network.GetOutputs(), kTargetValues);
    derivative_values = m_neural_network.GetLayer(layers_number-1).CalculateDerivativeValues();
    local_gradients = error * derivative_values;    //Out layer

    for (int it = layers_number-1; it > 0; --it)    //Hidden layers
    {
        derivative_values = m_neural_network.GetLayer(it - 1).CalculateDerivativeValues();

        weights = m_neural_network.GetLayer(it).GetSynapticWeights();
        
        last_delta_weights = m_neural_network.GetLayer(it).GetDeltaWeigths();
        input_vector = m_neural_network.GetLayer(it-1).GetActivatedValues();
        delta_weights = CalculateDeltaWeights(m_learning_rate, m_momentum, local_gradients, input_vector, last_delta_weights);
        m_neural_network.GetLayer(it).AdjustmentWeights(delta_weights);

        last_delata_biases = m_neural_network.GetLayer(it).GetBiases();
        delta_biases = CalculateDeltaBiases(m_learning_rate, m_momentum, local_gradients, last_delata_biases);
        m_neural_network.GetLayer(it).AdjustmentBiases(delta_biases);

        local_gradients = CalculateLocalGradients(derivative_values, local_gradients, weights);
    }

    //Input layer
    weights = m_neural_network.GetLayer(0).GetSynapticWeights();
    last_delta_weights = m_neural_network.GetLayer(0).GetDeltaWeigths();
    delta_weights = CalculateDeltaWeights(m_learning_rate, m_momentum, local_gradients, kInputData, last_delta_weights);
    m_neural_network.GetLayer(0).AdjustmentWeights(delta_weights);
    last_delata_biases = m_neural_network.GetLayer(0).GetBiases();
    delta_biases = CalculateDeltaBiases(m_learning_rate, m_momentum, local_gradients, last_delata_biases);
    m_neural_network.GetLayer(0).AdjustmentBiases(delta_biases);
}

Vector<double> BackPropagation::GetError()
{
    return m_error;
}

NeuralNetwork BackPropagation::GetNeuralNetwork() const
{
    return m_neural_network;
}
