#pragma once

#include "pch.h"
#include "neural_network.h"

class BackPropagation
{
private:
    NeuralNetwork m_neural_network;
    double m_learning_rate;
    double m_momentum;
public:
    BackPropagation() = delete;
    BackPropagation(
        const NeuralNetwork& kNeuralNetwork,
        const double kLearningRate, const double kMomentum) :
        m_neural_network(kNeuralNetwork), m_learning_rate(kLearningRate),
        m_momentum(kMomentum) {};

    Vector<double> CalculateError(
        const Vector<double>& kOutputValues, 
        const Vector<double>& kTargetValues
    );
    
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

    void AdjustmentWeight(
        const Vector<double>& kInputData,
        const Vector<double>& kTargetValues
    );
};