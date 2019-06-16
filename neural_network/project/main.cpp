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

	}
	catch (err::NNException& e)
	{
		
	}
	catch (MatrixException& me)
	{
		std::cout << me.what();
	}

    return 0;
}
