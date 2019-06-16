#pragma once

#include "layer.h"
#include <gtest/gtest.h>

Layer CreateTestLayer()
{
	srand(0);
	const size_t kInputsNumber = 2;
	const size_t kNeuronsNumber = 2;
	Layer layer(kInputsNumber, kNeuronsNumber, LINEAR);
	layer.SetSynapticWeights(Matrix<double>(2, 2, { 1,2,3,4 }));
	layer.SetBiases({ 1,1 });
	return layer;
}

TEST(Layer_tests, t0_random_initialisation)
{
	Layer layer = CreateTestLayer();
	double treshold = 10;
	layer.InitializeRandomWeights(-treshold, treshold);
	layer.InitializeRandomBiases(-treshold, treshold);
	
	auto weights = layer.GetSynapticWeights().GetVector();
	auto biases = layer.GetBiases().GetVector();

	std::for_each(weights.begin(), weights.end(), [&](double n) { ASSERT_TRUE((n > -treshold) && (n < treshold)); });
	std::for_each(biases.begin(), biases.end(), [&](double n) { ASSERT_TRUE((n > -treshold) && (n < treshold)); });
}


TEST(Layer_tests, t1_calculate_local_field)
{
	Layer layer = CreateTestLayer();
	Vector<double> input = { 1,2 };
	Vector<double> local_field_actual = layer.CalculateLocalFields(input);
	Vector<double> local_field_expect{ 6,12 };
	for (size_t it = 0; it < layer.GetNeuronsNumber(); ++it)
	{
		ASSERT_EQ(local_field_actual.at(it), local_field_expect.at(it));
	}
}


TEST(Layer_tests, t2_calculate_activated_values)
{
	Layer layer = CreateTestLayer();

	Vector<double> input = { 1,2 };
	Vector<double> activated_values_actual = layer.CalculateLocalFields(input);
	Vector<double> activated_values_expect{ 6,12 };
	for (size_t it = 0; it < layer.GetNeuronsNumber(); ++it)
	{
		ASSERT_EQ(activated_values_actual.at(it), activated_values_expect.at(it));
	}
}

TEST(Layer_tests, t3_calculate_derivative_values)
{
	Layer layer = CreateTestLayer();
	Vector<double> input = { 1,2 };
	Vector<double> derivative_value_actual = layer.CalculateDerivativeValues(input);
	Vector<double> derivative_value_expect{ 1, 1};
	for (size_t it = 0; it < layer.GetNeuronsNumber(); ++it)
	{
		ASSERT_EQ(derivative_value_actual.at(it), derivative_value_expect.at(it));
	}
}

TEST(Layer_tests, t4_adjustment_weights)
{
	Layer layer = CreateTestLayer();
	Matrix<double> delta_weigths = -1.0*layer.GetSynapticWeights();
	layer.AdjustmentWeights(delta_weigths);
	auto new_weights = layer.GetSynapticWeights();
	for (size_t it = 0; it < new_weights.GetSize(); ++it)
	{
		ASSERT_EQ(new_weights.at(it), 0);
	}
}

TEST(Layer_tests, t5_adjustment_biases)
{
	Layer layer = CreateTestLayer();
	Vector<double> delta_biases = -1.0 * layer.GetBiases();
	layer.AdjustmentBiases(delta_biases);
	auto new_biases = layer.GetBiases();
	for (size_t it = 0; it < new_biases.GetSize(); ++it)
	{
		ASSERT_EQ(new_biases.at(it), 0);
	}
}