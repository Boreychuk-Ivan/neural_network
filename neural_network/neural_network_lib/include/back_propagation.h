#pragma once

#include "pch.h"
#include "neural_network.h"

class BackPropagation
{
public:
    BackPropagation() = delete;
    BackPropagation(
        NeuralNetwork& kNeuralNetwork,
        const double kLearningRate, const double kMomentum);
    
    Vector<double> CalculateLocalGradients
    (
        const Vector<double>& kDerivativeValues, 
        const Vector<double>& kPreviousLayerError, 
        const Matrix<double>& kSynapticWeights
    );

    Matrix<double> CalculateDeltaWeights
    (
        const double& kLearningRate, 
        const double& kMomentum,
        const Vector<double>& kLocalGradients, 
        const Vector<double>& kInputValues,
        const Matrix<double>& kLastDeltaWeigths
    );

    Vector<double> CalculateDeltaBiases
    (
        const double& kLearningRate,
        const double& kMomentum,
        const Vector<double>& kLocalGradients,
        const Vector<double>& kLastDeltaBiases
    );

    void AdjustmentNeuralNetwork(
        const Vector<double>& kInputData,
        const Vector<double>& kTargetValues
    );

    NeuralNetwork GetNeuralNetwork() const;
    Matrix<double> GetDeltaWeights(const size_t kLayerNum) const;
    Vector<double> GetDeltaBiases(const size_t kLayerNum) const;

private:
    NeuralNetwork& m_neural_network;
    double m_learning_rate;
    double m_momentum;

    std::vector<Matrix<double>> m_delta_weights;
    std::vector<Vector<double>> m_delta_biases;

};