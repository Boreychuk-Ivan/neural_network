#pragma once
#include "pch.h"
#include "neural_network.h"

class FileHander
{
public:
    void WriteNeuralNetworkToFile(const std::string& kFilePath, const NeuralNetwork& kNeuralNetwork);
    NeuralNetwork ReadNeuralNetworkFromFile(const std::string& kFilePath);
    Matrix<double> ReadMatrixFromFile(std::ifstream* file, std::streampos pos, const size_t& kRows, const size_t& kCols);
    std::vector<unsigned> ReadVectorFromFile(std::ifstream* file, std::streampos pos);
};
