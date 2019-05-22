#include "file_handler.h"

void FileHander::WriteNeuralNetworkToFile(const std::string& kFilePath, const NeuralNetwork & kNeuralNetwork)
{
    std::ofstream out(kFilePath, std::ios::out || std::ios::trunc);
    if (!out.is_open())
    {
        std::cerr << "Error! File was not opened!\n";
        exit(1);
    }

    out << "Neural network:" << kNeuralNetwork.GetLayersNumber() << "\n";
    for (size_t it = 0; it < kNeuralNetwork.GetLayersNumber(); ++it)
    {
        Layer current_layer = kNeuralNetwork.GetLayer(it);
        out << "# Layer(inputs,neurons):"
            << current_layer.GetInputsNumber()
            << current_layer.GetNeuronsNumber() << std::endl;

        out << "## Weights: \n" << current_layer.GetSynapticWeights();
        out << "## Biases: \n" << current_layer.GetBiases();
        out << "\n";
    }
}
