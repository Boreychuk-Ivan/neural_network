#include "pch.h"

//Tests
#include "matrix_tests.h"
#include "neuron_tests.h"


int main(int argc, char* argv[])
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
