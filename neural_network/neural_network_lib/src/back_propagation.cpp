#include "back_propagation.h"

Vector<double> BackPropagation::CalculateError(const Vector<double> kOutputValues, const Vector<double> kTargetValues)
{
    assert(kOutputValues.IsEqualSize(kTargetValues));
    return Vector<double>(kTargetValues - kOutputValues);
}

Vector<double> BackPropagation::CalculateLocalGradients(
    const Vector<double> kDerivativeValues, 
    const Vector<double> kPreviousLayerError, 
    const Matrix<double> kSynapticWeights)
{
    return Vector<double>(kDerivativeValues.dot(kPreviousLayerError * kSynapticWeights));
}

Matrix<double> BackPropagation::CalculateDeltaWeights(
    const double kLearningRate, 
    const Vector<double> kLocalGradients, 
    const Vector<double> kInputValues)
{
    
}
