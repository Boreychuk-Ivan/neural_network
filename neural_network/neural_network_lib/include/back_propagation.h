#pragma once

#include "pch.h"
#include "neural_network.h"

class BackPropagation
{
private:
    NeuralNetwork m_neural_network;
public:
    Vector<double> CalculateError(const Vector<double>& kOutputValues, const Vector<double>& kTargetValues);
    
    Vector<double> CalculateLocalGradients(
        const Vector<double>& kDerivativeValues, 
        const Vector<double>& kPreviousLayerError, 
        const Matrix<double>& kSynapticWeights
    );

    Matrix<double> CalculateDeltaWeights(
        const double& kLearningRate, 
        const double& kMomentum,
        const Vector<double>& kLocalGradients, 
        const Vector<double>& kInputValues,
        const Matrix<double>& kLastDeltaWeigths
    );

    void AdjustmentWeight();
};