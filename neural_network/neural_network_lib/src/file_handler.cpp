#include "file_handler.h"

void FileHander::WriteNeuralNetworkToFile(const std::string& kFilePath, const NeuralNetwork & kNeuralNetwork)
{
    std::ofstream out(kFilePath, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        std::cerr << "Error! File was not opened!\n";
        exit(1);
    }

    out << "Neural network(layers):" << kNeuralNetwork.GetLayersNumber() << "\n";
    for (size_t it = 0; it < kNeuralNetwork.GetLayersNumber(); ++it)
    {
        Layer current_layer = kNeuralNetwork.GetLayer(it);
        out << "# Layer(inputs,neurons):"
            << current_layer.GetInputsNumber()
            << current_layer.GetNeuronsNumber() << std::endl;

        out << "## Weights:" << current_layer.GetSynapticWeights();
        out << "## Biases:" << current_layer.GetBiases();
        out << "\n";
    }
    std::cout << "File with neural network was written!\n";
    out.close();
}

//NeuralNetwork FileHander::ReadNeuralNetworkFromFile(const std::string & kFilePath)
//{
//    
//}

Matrix<double> FileHander::ReadMatrixFromFile(std::ifstream* file, std::streampos pos, const size_t & kRows, const size_t & kCols)
{
    file->seekg(pos);
    Matrix<double> read_mtx(kRows, kCols);
    size_t row_it = 0;
    size_t col_it = 0;
    while (row_it < kRows)
    {
        if (isdigit(file->peek()))
        {
            double read_value;
            *file >> read_value;
            read_mtx.at(row_it, col_it) = read_value;
            col_it++;
            if (col_it == kCols)
            {
                col_it = 0;
                row_it++;
            }
        }
        else
            file->get();
    }
    return read_mtx;
}
