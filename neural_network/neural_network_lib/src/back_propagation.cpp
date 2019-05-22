#include "back_propagation.h"

Vector<double> BackPropagation::CalculateError(
    const Vector<double>& kOutputValues, 
    const Vector<double>& kTargetValues)
{
    assert(kOutputValues.IsEqualSize(kTargetValues));
    return Vector<double>(kTargetValues - kOutputValues);
}

Vector<double> BackPropagation::CalculateLocalGradients(
    const Vector<double>& kDerivativeValues, 
    const Vector<double>& kPreviousLayerError, 
    const Matrix<double>& kSynapticWeights)
{
    return Vector<double>(kDerivativeValues.DotMult(kPreviousLayerError * kSynapticWeights));
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

void BackPropagation::AdjustmentWeight(const Vector<double>& kInputData, const Vector<double>& kTargetValues)
{
   size_t layers_number = m_neural_network.GetLayersNumber();
   Matrix<double> weights;
   weights.InitialiseDiag(m_neural_network.GetOutputsNumber());
   Vector<double> out_error;
   Vector<double> derivative_values;
   Vector<double> local_gradients;
   Vector<double> input_vector;
   Vector<double> last_delta_weights;
   Matrix<double> delta_weights;
   
   for (int it = layers_number - 1; it >= 0; --it)
   {
       if(it == layers_number - 1)  //Output layer
       {
           weights.InitialiseDiag(m_neural_network.GetOutputsNumber());
           local_gradients = CalculateError(m_neural_network.GetOutputs(), kTargetValues);
       }
       else
       {
           weights = m_neural_network.GetLayer(it).GetSynapticWeights();
       }
       
       derivative_values = m_neural_network.GetLayer(it).CalculateDerivativeValues();
       local_gradients = CalculateLocalGradients(derivative_values, local_gradients, weights);
       last_delta_weights = m_neural_network.GetLayer(it).GetDeltaWeigths();
       input_vector = (it != 0) ? m_neural_network.GetLayer(it - 1).GetActivatedValues() : kInputData;
       delta_weights = CalculateDeltaWeights(m_learning_rate, m_momentum, local_gradients, input_vector, last_delta_weights);
       m_neural_network.GetLayer(it).AdjustmentWeight(delta_weights);
   }
}
