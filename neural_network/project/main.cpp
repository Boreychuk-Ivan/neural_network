#include "pch.h"
#include "matrix_lib.h"
#include "neural_network.h"
#include "file_handler.h"
#include "training.h"
#include "back_propagation.h"
#include "loss_functions.h"

int main()
{
    srand(1);
    NeuralNetwork nn_test({ 2, 2, 1 });
    Training trainer(nn_test, 0.5, 0.00001, "../../data/training_set.dt", 2, 1, 4);
    trainer.TrainNeuralNetwork(10000);

    return 0;
}
