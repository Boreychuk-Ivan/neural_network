#include "pch.h"
#include "matrix_lib.h"
#include "neural_network.h"
#include "file_handler.h"
#include "training.h"

int main()
{
    srand(15);
    //Create NN
    NeuralNetwork nn_test({2, 5, 1});
    nn_test.SetActivationFunction(0, LINEAR);
    nn_test.SetActivationFunction(1, LINEAR);
    


    Training trainer(nn_test, 0.5, 0.1, "../../data/training_set.dt", 2, 1, 4);
    trainer.TrainNeuralNetwork(1000);

    return 0;
}
