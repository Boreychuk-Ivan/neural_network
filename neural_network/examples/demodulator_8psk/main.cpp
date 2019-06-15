#include "pch.h"
#include "neural_network.h"
#include "training.h"
#include "file_handler.h"

int main()
{
	srand(1);
	// Neural network creating: {inputs, hidden neurons, outputs}:
	const size_t kInputsNumber = 2;
	const size_t kOutputsNumber = 8;
	const size_t kHiddenNeurons = 8;
	NeuralNetwork neural_network({ kInputsNumber, kHiddenNeurons, kOutputsNumber });

	//Trainer creation
	const double kLearningRate = 0.7;
	const double kMomentum = 0.0001;

	std::string training_file = "data/training_set.dt";
	const size_t kTrainingSetSize = 8;
	Training trainer(neural_network, kLearningRate, kMomentum, training_file, kInputsNumber, kOutputsNumber, kTrainingSetSize);

	//Train neural network 
	const size_t kEpochNumber = 2000;
	trainer.TrainNeuralNetwork(kEpochNumber);

	//Save neural network 
	std::string file_neural_network = "data/neural_network.dt";
	FileHander fh;
	fh.WriteNeuralNetworkToFile(file_neural_network, neural_network);

	//Read neural network from file
	NeuralNetwork neural_network_learned = fh.ReadNeuralNetworkFromFile(file_neural_network);

	//Test neural network
	auto ResultToNumber = [](const Vector<double>& kResultVector)->int
	{
		int result = -1;
		double threshold = 0.85;
		for (size_t it = 0; it < kResultVector.GetSize(); ++it)
		{
			if (kResultVector.at(it) > threshold) result = it;
		}
		return result;
	};

	std::cout << ".......... Test neural network ..........\n";
	std::cout << "Inputs : 0 -  0.9239+0.3827i" << ", Output: " << ResultToNumber(neural_network_learned.CalculateOutputs({0.9239, 0.3827}))    << "\n";
	std::cout << "Inputs : 1 -  0.3827+0.9239i" << ", Output: " << ResultToNumber(neural_network_learned.CalculateOutputs({0.3827, 0.9239}))    << "\n";
	std::cout << "Inputs : 2 - -0.3827+0.9239i" << ", Output: " << ResultToNumber(neural_network_learned.CalculateOutputs({-0.3827, 0.9239 }))  << "\n";
	std::cout << "Inputs : 3 - -0.9239+0.3827i" << ", Output: " << ResultToNumber(neural_network_learned.CalculateOutputs({-0.9239, 0.3827 }))  << "\n";
	std::cout << "Inputs : 4 - -0.9239-0.3827i" << ", Output: " << ResultToNumber(neural_network_learned.CalculateOutputs({-0.9239, -0.3827 })) << "\n";
	std::cout << "Inputs : 5 - -0.3827-0.9239i" << ", Output: " << ResultToNumber(neural_network_learned.CalculateOutputs({-0.3827, -0.9239 })) << "\n";
	std::cout << "Inputs : 6 -  0.3827-0.9239i" << ", Output: " << ResultToNumber(neural_network_learned.CalculateOutputs({0.3827, -0.9239 }))  << "\n";
	std::cout << "Inputs : 7 -  0.9239-0.3827i" << ", Output: " << ResultToNumber(neural_network_learned.CalculateOutputs({0.9239, -0.3827 }))  << "\n";

	return 0;
}