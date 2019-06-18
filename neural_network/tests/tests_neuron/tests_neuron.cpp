#pragma once
#include "neuron.h"
#include <gtest/gtest.h>

TEST(Neuron_tests, t0_calculate_activated_value)
{
	Neuron neuron_0(0, SIGMOID);
	Neuron neuron_1(0, TANH);
	Neuron neuron_2(0, LINEAR);

	double input = 0.2;
	double output_0_actual = neuron_0.CalculateActivatedValue(input);
	double output_1_actual = neuron_1.CalculateActivatedValue(input);
	double output_2_actual = neuron_2.CalculateActivatedValue(input);

	double output_0_expect = 0.5498339;
	double output_1_expect = 0.1973753;
	double output_2_expect = 0.2;

	ASSERT_TRUE((output_0_actual - output_0_expect) < 1e-3);
	ASSERT_TRUE((output_1_actual - output_1_expect) < 1e-3);
	ASSERT_TRUE((output_2_actual - output_2_expect) < 1e-3);
}

TEST(Neuron_tests, t0_calculate_derivative_value)
{
	Neuron neuron_0(0, SIGMOID);
	Neuron neuron_1(0, TANH);
	Neuron neuron_2(0, LINEAR);

	double input = 0.2;
	double derivative_0_actial = neuron_0.CalculateDerivativeValue(input);
	double derivative_1_actial = neuron_1.CalculateDerivativeValue(input);
	double derivative_2_actial = neuron_2.CalculateDerivativeValue(input);

	double derivative_0_expect = 0.24751657;
	double derivative_1_expect = 0.96104298;
	double derivative_2_expect = 1.0;

	ASSERT_TRUE((derivative_0_actial - derivative_0_expect) < 1e-3);
	ASSERT_TRUE((derivative_1_actial - derivative_1_expect) < 1e-3);
	ASSERT_TRUE((derivative_2_actial - derivative_2_expect) < 1e-3);
}

int main(int argc, char* argv[])
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
