// Auxillary methods
#pragma once
#include "pch.h"
#include "layer.h"
#include "neural_network.h"
#include "back_propagation.h"


namespace aux
{
    template <class T>
    bool CompareVector(const std::vector<T>& kV_0, const std::vector<T>& kV_1)
    {
      if (kV_0.size() != kV_1.size()) return 0;
      for (size_t it = 0; it < kV_0.size(); ++it)
      {
        if (std::abs(kV_0.at(it) - kV_1.at(it)) > 1e-3) return 0;
      }
      return 1;
    }


	Layer CreateTestLayer()
	{
		srand(0);
		const size_t kInputsNumber = 2;
		const size_t kNeuronsNumber = 2;
		Layer layer(kInputsNumber, kNeuronsNumber, LINEAR);
		layer.SetSynapticWeights(Matrix<double>(2, 2, { 1,2,3,4 }));
		layer.SetBiases({ 1,1 });
		return layer;
	}


	NeuralNetwork CreateTestNeuralNetwork()
	{
		const int kInputsNumber = 2;
		const int kOutputsNumber = 2;
		const int kHiddenNeurons = 2;
		NeuralNetwork neural_network({ kInputsNumber, kHiddenNeurons, kOutputsNumber });
		neural_network.SetActivationFunction(0, LINEAR);
		neural_network.SetActivationFunction(1, LINEAR);
		neural_network.SetSynapticWeigths(0, Matrix<double>(2, 2, { 1,2,3,4 }));
		neural_network.SetSynapticWeigths(1, Matrix<double>(2, 2, { 1,2,3,4 }));
		neural_network.SetBiases(0, { 1,1 });
		neural_network.SetBiases(1, { 1,1 });
		return neural_network;
	}

	BackPropagation CreateBackPropagationObj()
	{
		const double kLearningRate = 0.5;
		const double kMomentum = 0.01;
		return BackPropagation(CreateTestNeuralNetwork(), kLearningRate, kMomentum);
	}
}
