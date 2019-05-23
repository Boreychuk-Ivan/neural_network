#include "pch.h"
#include "matrix_lib.h"
#include "neural_network.h"
#include "file_handler.h"

int main()
{
    srand(1);
    NeuralNetwork nn_test({2,2,2});

    nn_test.SetActivationFunction(0, LINEAR);
    nn_test.SetActivationFunction(1, LINEAR);
    double weights[4] = { 1,2,3,4 };
    nn_test.SetSynapticWeigths(0,Matrix<double>(2,2, weights));
    nn_test.SetSynapticWeigths(1, Matrix<double>(2, 2, weights));
    nn_test.SetBiases(0, { 0,0 });
    nn_test.SetBiases(1, { 0,0 });
   // nn_test.DisplayArchitecture();

    Vector<double> input_vector{ 1,2 };
    nn_test.PushInputData(input_vector);
    nn_test.CalculateOutputs();
    //nn_test.DisplayLayers();
    //nn_test.DisplayNeurons();

    FileHander fh;
    fh.WriteNeuralNetworkToFile("../neural_network.dt", nn_test);

    NeuralNetwork read_nn = fh.ReadNeuralNetworkFromFile("../neural_network.dt");
    read_nn.DisplayArchitecture();
    read_nn.DisplayLayers();

    //std::ifstream mtx_file("../test_matrix.dt", std::ios::in);
    //std::cout << fh.ReadMatrixFromFile(&mtx_file, 0, 3,3);
    
    return 0;
}
