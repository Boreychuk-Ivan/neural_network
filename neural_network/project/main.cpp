#include "pch.h"
#include "matrix_lib.h"
#include "neural_network.h"
#include "file_handler.h"
#include "training.h"
#include "back_propagation.h"
#include "loss_functions.h"

int main()
{
	try
	{
		srand(1);
		Neuron n(0, SIGMOID);
	}
	catch (err::NNException& e)
	{
		std::cout << e.what();
	}

    return 0;
}
