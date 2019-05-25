#include "pch.h"
#include "matrix_lib.h"
#include "neural_network.h"
#include "file_handler.h"
#include "training.h"
#include "back_propagation.h"

int main()
{
    srand(11); //11
    //Create NN
    NeuralNetwork nn_test({2, 2, 1});

    //HABR
    //nn_test.SetSynapticWeigths(0, Matrix<double>(2, 2, { 0.45,-0.12,0.78,0.13 }));
    //nn_test.SetSynapticWeigths(1, Matrix<double>(1, 2, { 1.5, -2.3 }));
    //nn_test.SetBiases(0, {0,0});
    //nn_test.SetBiases(1, { 0 });

   // nn_test.DisplayLayers();

    Vector<double> input_vector{ 1,0 };
    //std::cout << "OUT : " << nn_test.CalculateOutputs(input_vector) << "\n";
  //  nn_test.DisplayNeurons();

   // std::cout << "CORRECTION!!!\n";
    BackPropagation bp(nn_test, 0.8, 0.00001);

    for(int it = 0; it < 5000; ++it)
    {
       // std::cout << "*********************************\n";
        Vector<double> input_vector{ 0,0 };
        bp.AdjustmentNeuralNetwork(input_vector, { 0 });
        //std::cout << "OUT : " << nn_test.CalculateOutputs(input_vector) << "\n";

        input_vector = { 0,1 };
        bp.AdjustmentNeuralNetwork(input_vector, { 0 });
        //std::cout << "OUT : " << nn_test.CalculateOutputs(input_vector) << "\n";

        input_vector = { 1,0 };
        bp.AdjustmentNeuralNetwork(input_vector, { 0 });
        //std::cout << "OUT : " << nn_test.CalculateOutputs(input_vector) << "\n";

        input_vector = { 1,1 };
        bp.AdjustmentNeuralNetwork(input_vector, { 1 });
        //std::cout << "OUT : " << nn_test.CalculateOutputs(input_vector) << "\n";
    }
    
    std::cout << "OUT after learning: " << nn_test.CalculateOutputs({ 1,0 }) << "\n";
    std::cout << "OUT after learning: " << nn_test.CalculateOutputs({ 1,1 }) << "\n";
    std::cout << "OUT after learning: " << nn_test.CalculateOutputs({ 0,0 }) << "\n";
    std::cout << "OUT after learning: " << nn_test.CalculateOutputs({ 0,1 }) << "\n";
    //nn_test.DisplayNeurons();

    //std::cout << "Layer # 1 : \n" << bp.GetDeltaWeights(1) << "\n";
    //std::cout << "Layer # 0 : \n" << bp.GetDeltaWeights(0) << "\n";
    //nn_test.DisplayLayers();

    //Training trainer(nn_test, 0.7, 0.3, "../../data/training_set.dt", 2, 1, 4);
    //trainer.TrainNeuralNetwork(10);

    return 0;
}
