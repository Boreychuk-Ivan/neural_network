#include "pch.h"
#include "neural_network.h"
#include "training.h"
#include "file_handler.h"

int main()
{
	srand(1);
	// Neural network creating: {inputs, hidden neurons, outputs}:
	const size_t kInputsNumber = 2;
	const size_t kOutputsNumber = 1;
	const size_t kHiddenNeurons = 2;
	NeuralNetwork neural_network({ kInputsNumber, kHiddenNeurons, kOutputsNumber });
	
	//Trainer creation
	const double kLearningRate = 0.7;
	const double kMomentum = 0;
	
	std::string training_file = "data/training_set.dt";
	const size_t kTrainingSetSize = 4;
	Training trainer(neural_network, kLearningRate, kMomentum, training_file, kInputsNumber, kOutputsNumber, kTrainingSetSize);

	//Train neural network 
	const size_t kEpochNumber = 5000;
	trainer.TrainNeuralNetwork(kEpochNumber);

	//Save neural network 
	std::string file_neural_network = "data/neural_network.dt";
	FileHander fh;
	fh.WriteNeuralNetworkToFile(file_neural_network, neural_network);

	//Read neural network from file
	NeuralNetwork neural_network_learned = fh.ReadNeuralNetworkFromFile(file_neural_network);

	//Test neural network
	std::cout << ".......... Test neural network ..........\n";
	std::cout << "Inputs : 0 0" << ", Output: " << neural_network_learned.CalculateOutputs({0,0});
	std::cout << "Inputs : 0 1" << ", Output: " << neural_network_learned.CalculateOutputs({0,1});
	std::cout << "Inputs : 1 0" << ", Output: " << neural_network_learned.CalculateOutputs({1,0});
	std::cout << "Inputs : 1 1" << ", Output: " << neural_network_learned.CalculateOutputs({1,1});

	return 0;
}