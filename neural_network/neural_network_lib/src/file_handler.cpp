#include "file_handler.h"

void FileHander::WriteNeuralNetworkToFile(const std::string& kFilePath, const NeuralNetwork & kNeuralNetwork)
{
    std::ofstream out(kFilePath, std::ios::out | std::ios::trunc);
	err::assert_throw(out.is_open(), "Error <WriteNeuralNetworkToFile>: " + kFilePath + "was not opened for write!\n");

    out << "Neural network(layers):";
    out << kNeuralNetwork.GetLayer(0).GetInputsNumber();
    for (size_t it = 0; it < kNeuralNetwork.GetLayersNumber(); ++it)
    {
        out << "," << kNeuralNetwork.GetLayer(it).GetNeuronsNumber();
    }
    out << std::endl;

    for (size_t it = 0; it < kNeuralNetwork.GetLayersNumber(); ++it)
    {
        Layer current_layer = kNeuralNetwork.GetLayer(it);
        out << "# Layer(inputs,neurons):"
            << current_layer.GetInputsNumber() << ","
            << current_layer.GetNeuronsNumber() << std::endl;
        out << "## Activation function:" << 
            ActivationFunctions::GetString(current_layer.GetActivationFunctionType()) << std::endl;
        out << "## Weights:" << current_layer.GetSynapticWeights();
        out << "## Biases:" << !current_layer.GetBiases();
        out << "\n";
    }
    std::cout << "File with neural network was written!\n\n";
    out.close();
}

NeuralNetwork FileHander::ReadNeuralNetworkFromFile(const std::string & kFilePath)
{
    std::ifstream file(kFilePath, std::ios::in);
	err::assert_throw(file.is_open(), "Error <ReadNeuralNetworkFromFile>: " + kFilePath + "was not opened for write!\n");

    std::string input_line;
    std::getline(file, input_line, ':');
    err::assert_throw(input_line == "Neural network(layers)", "Error <ReadNeuralNetworkFromFile>: reading problem\n");
    std::vector<int> architecture = ReadVectorFromFile(&file, file.tellg());
    file.get(); //'\n'
    NeuralNetwork neureal_network(architecture);
    for (size_t it = 0; it < architecture.size()-1; ++it)
    {
        std::getline(file, input_line, ':'); 
		err::assert_throw(input_line == "# Layer(inputs,neurons)", "Error <ReadNeuralNetworkFromFile>: read problem\n");
        std::vector<int> layer_params = ReadVectorFromFile(&file, file.tellg());
        file.get(); //'\n'
        
        std::getline(file, input_line, ':');
		err::assert_throw(input_line == "## Activation function", "Error <ReadNeuralNetworkFromFile>: read problem\n");
        std::getline(file, input_line, '\n');
        neureal_network.SetActivationFunction(it, ActivationFunctions::Type(input_line));

        std::getline(file, input_line, ':');
		err::assert_throw(input_line == "## Weights", "Error <ReadNeuralNetworkFromFile>: read problem\n");
        file.get(); 
        Matrix<double> weigths = ReadMatrixFromFile(&file, file.tellg(), layer_params.at(1), layer_params.at(0));
        neureal_network.SetSynapticWeigths(it, weigths);
        file.get(); file.get(); 

        std::getline(file, input_line, ':');
		err::assert_throw(input_line == "## Biases", "Error <ReadNeuralNetworkFromFile>: read problem\n");
        file.get(); 
        Matrix<double> biases = ReadMatrixFromFile(&file, file.tellg(), layer_params.at(1), 1);
        biases = !biases;
        neureal_network.SetBiases(it, biases);
        file.get(); file.get(); file.get();
    }
    return neureal_network;
}


Matrix<double> FileHander::ReadMatrixFromFile(std::ifstream* file, std::streampos pos, const size_t & kRows, const size_t & kCols)
{
	err::assert_throw(file->is_open(), "Error <ReadMatrixFromFile> : file was not opened\n");
    file->seekg(pos);
    Matrix<double> read_mtx((int)kRows, (int)kCols);
    size_t row_it = 0;
    size_t col_it = 0;
    while (row_it < kRows)
    {
        if (isdigit(file->peek()) || file->peek() == '-')
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

std::vector<int> FileHander::ReadVectorFromFile(std::ifstream * file, std::streampos pos)
{
    if(file->tellg() != pos) file->seekg(pos);

    std::vector<int> return_vector;
    while (file->peek() != '\n')
    {
        if (isdigit(file->peek()))
        {
            size_t neurons_num;
            *file >> neurons_num;
            return_vector.push_back((int)neurons_num);
        }
        else
        {
            file->get();
        }
    }
    return return_vector;
}
