#include "pch.h"
#include "matrix_lib.h"
#include "neural_network.h"
#include "training.h"

int main()
{
    //srand(1);
    //NeuralNetwork nn_test({2,2,2});

    //nn_test.SetActivationFunction(0, LINEAR);
    //nn_test.SetActivationFunction(1, LINEAR);
    //double weights[4] = { 1,2,3,4 };
    //nn_test.SetSynapticWeigths(0,Matrix<double>(2,2, weights));
    //nn_test.SetSynapticWeigths(1, Matrix<double>(2, 2, weights));
    //nn_test.SetBiases(0, { 0,0 });
    //nn_test.SetBiases(1, { 0,0 });
    //nn_test.DisplayArchitecture();

    //Vector<double> input_vector{ 1,2 };
    //nn_test.PushInputData(input_vector);
    //nn_test.CalculateOutputs();
    //nn_test.DisplayLayers();
    //nn_test.DisplayNeurons();
    
    std::string file_path("../../data/training_set.dt");
    Training train_object(file_path, 2, 1, 3);
    std::cout << "Inputs:\n" << train_object.GetInputMatrix();
    std::cout << "Outputs: \n" << train_object.GetOutputMatrix();

    return 0;
}
