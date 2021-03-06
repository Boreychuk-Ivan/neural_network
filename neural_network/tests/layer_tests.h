#pragma once

#include "layer.h"
#include <gtest/gtest.h>
#include "auxillary_functions.h"


TEST(Layer_tests, t0_random_initialisation)
{
	Layer layer = aux::CreateTestLayer();
	double treshold = 10;
	layer.InitializeRandomWeights(-treshold, treshold);
	layer.InitializeRandomBiases(-treshold, treshold);
	
	auto weights = layer.GetSynapticWeights().GetVector();
	auto biases = layer.GetBiases().GetVector();

	std::for_each(weights.begin(), weights.end(), [&](double n) { ASSERT_TRUE((n > -treshold) && (n < treshold)); });
	std::for_each(biases.begin(), biases.end(), [&](double n) { ASSERT_TRUE((n > -treshold) && (n < treshold)); });

	ASSERT_THROW(layer.InitializeRandomWeights(treshold, -treshold), err::NNException);
	ASSERT_THROW(layer.InitializeRandomBiases(treshold, -treshold), err::NNException);
}


TEST(Layer_tests, t1_calculate_local_field)
{
	Layer layer = aux::CreateTestLayer();
	Vector<double> input = { 1,2 };
	Vector<double> local_field_actual = layer.CalculateLocalFields(input);
	Vector<double> local_field_expect{ 6,12 };
	for (size_t it = 0; it < layer.GetNeuronsNumber(); ++it)
	{
		ASSERT_EQ(local_field_actual.at(it), local_field_expect.at(it));
	}

	ASSERT_THROW(layer.CalculateLocalFields({}), err::NNException);
	ASSERT_THROW(layer.CalculateLocalFields({1,2,3,4}), err::NNException);
}


TEST(Layer_tests, t2_calculate_activated_values)
{
	Layer layer = aux::CreateTestLayer();

	Vector<double> input = { 1,2 };
	Vector<double> activated_values_actual = layer.CalculateActivatedValues(input);
	Vector<double> activated_values_expect{ 6,12 };
	for (size_t it = 0; it < layer.GetNeuronsNumber(); ++it)
	{
		ASSERT_EQ(activated_values_actual.at(it), activated_values_expect.at(it));
	}

	ASSERT_THROW(layer.CalculateActivatedValues({}), err::NNException);
	ASSERT_THROW(layer.CalculateActivatedValues({ 1,2,3,4 }), err::NNException);
}

TEST(Layer_tests, t3_calculate_derivative_values)
{
	Layer layer = aux::CreateTestLayer();
	Vector<double> input = { 1,2 };
	Vector<double> derivative_value_actual = layer.CalculateDerivativeValues(input);
	Vector<double> derivative_value_expect{ 1, 1};
	for (size_t it = 0; it < layer.GetNeuronsNumber(); ++it)
	{
		ASSERT_EQ(derivative_value_actual.at(it), derivative_value_expect.at(it));
	}

	ASSERT_THROW(layer.CalculateDerivativeValues({}), err::NNException);
	ASSERT_THROW(layer.CalculateDerivativeValues({ 1,2,3,4 }), err::NNException);
}

TEST(Layer_tests, t4_adjustment_weights)
{
	Layer layer = aux::CreateTestLayer();
	Matrix<double> delta_weigths = -1.0*layer.GetSynapticWeights();
	layer.AdjustmentWeights(delta_weigths);
	auto new_weights = layer.GetSynapticWeights();
	for (size_t it = 0; it < new_weights.GetSize(); ++it)
	{
		ASSERT_EQ(new_weights.at(it), 0);
	}

	ASSERT_THROW(layer.AdjustmentWeights({}), MatrixException);
	ASSERT_THROW(layer.AdjustmentWeights(Matrix<double>(4,4)), MatrixException);
}

TEST(Layer_tests, t5_adjustment_biases)
{
	Layer layer = aux::CreateTestLayer();
	Vector<double> delta_biases = -1.0 * layer.GetBiases();
	layer.AdjustmentBiases(delta_biases);
	auto new_biases = layer.GetBiases();
	for (size_t it = 0; it < new_biases.GetSize(); ++it)
	{
		ASSERT_EQ(new_biases.at(it), 0);
	}

	ASSERT_THROW(layer.AdjustmentBiases({}), MatrixException);
	ASSERT_THROW(layer.AdjustmentBiases({1,2,3,4}), MatrixException);
}