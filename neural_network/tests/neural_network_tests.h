#pragma once
#include "auxillary_functions.h"
#include "neural_network.h"
#include <gtest/gtest.h>


TEST(Neural_network_tests, t0_constructor)
{
	ASSERT_THROW(NeuralNetwork({ 0 }), err::NNException);
	ASSERT_THROW(NeuralNetwork({0,-1, -2}), err::NNException);
}

TEST(Neural_network_tests, t1_calculate_outputs)
{
	NeuralNetwork neural_network = aux::CreateTestNeuralNetwork();
	Vector<double> inputs{ 1,1 };
	auto output_values_actual = neural_network.CalculateOutputs(inputs);
	Vector<double> output_values_expect{ 21,45 };
	ASSERT_TRUE(aux::CompareVector<double>(output_values_actual.GetVector(), output_values_expect.GetVector()));

	ASSERT_THROW(neural_network.CalculateOutputs({}), err::NNException);
	ASSERT_THROW(neural_network.CalculateOutputs({1,2,3,4}), err::NNException);
}

TEST(Neural_network_tests, t2_feed_forward)
{
	NeuralNetwork neural_network = aux::CreateTestNeuralNetwork();
	Vector<double> inputs{ 1,1 };
	auto output_values_actual = neural_network.FeedForward(inputs);
	Vector<double> output_values_expect{ 21,45 };
	ASSERT_TRUE(aux::CompareVector<double>(output_values_actual.GetVector(), output_values_expect.GetVector()));

	ASSERT_THROW(neural_network.FeedForward({}), err::NNException);
	ASSERT_THROW(neural_network.FeedForward({ 1,2,3,4 }), err::NNException);

	//Derivative values

	auto dvalues_layer0_actual = neural_network.GetLayer(0).GetDerivativeValues();
	auto dvalues_layer1_actual = neural_network.GetLayer(1).GetDerivativeValues();
	Vector<double> dvalues_layer0_expect{ 1,1 };
	Vector<double> dvalues_layer1_expect{ 1,1 };
	
	ASSERT_TRUE(aux::CompareVector<double>(dvalues_layer0_actual.GetVector(), dvalues_layer0_expect.GetVector()));
	ASSERT_TRUE(aux::CompareVector<double>(dvalues_layer1_actual.GetVector(), dvalues_layer1_expect.GetVector()));
}
