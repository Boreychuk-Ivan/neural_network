#pragma once

#include "pch.h"
#include "../auxillary_functions.h"
#include "back_propagation.h"
#include <gtest/gtest.h>

TEST(Back_propagation_tests, t0_calculate_local_gradient)
{
	BackPropagation bp = aux::CreateBackPropagationObj();

	const Vector<double> kDerivativeValues{1,1};
	const Vector<double> kPreviosLayerError{0.1, 0.1};
	const Matrix<double> kSynapticWeights(2, 2, {1,2,3,4});
	auto local_gradient_actual = bp.CalculateLocalGradients(kDerivativeValues, kPreviosLayerError, kSynapticWeights);
	std::vector<double> local_gradient_expect{ 0.4, 0.6 };
	
	ASSERT_TRUE(aux::CompareVector<double>(local_gradient_actual.GetVector(), local_gradient_expect));
	
	//Error check
	ASSERT_THROW(bp.CalculateLocalGradients(kDerivativeValues, kPreviosLayerError, {}), MatrixException);
	ASSERT_THROW(bp.CalculateLocalGradients(kDerivativeValues, {}, kSynapticWeights), MatrixException);
	ASSERT_THROW(bp.CalculateLocalGradients({}, kPreviosLayerError, kSynapticWeights), MatrixException);
}

TEST(Back_propagation_tests, t1_calculate_delta_weights)
{
	BackPropagation bp = aux::CreateBackPropagationObj();
	const double kLearningRate = 0.5;
	const double kMomentum = 0.5;
	const Vector<double> kLocalGradient{ 0.5,0.5 };
	const Vector<double> kInputValues{ 1,1 };
	const Matrix<double> kLastSynapticWeights(2, 2, {1,2,3,4});
	
	auto delta_weights_actual = bp.CalculateDeltaWeights(kLearningRate, kMomentum, kLocalGradient, kInputValues, kLastSynapticWeights);
	Matrix<double> delta_weights_expect(2, 2, { 0.75,1.25,  1.75,2.25 });

	ASSERT_TRUE(aux::CompareVector<double>(delta_weights_actual.GetVector(), delta_weights_expect.GetVector()));

	//Error check
	ASSERT_THROW(bp.CalculateDeltaWeights(kLearningRate, kMomentum, {}, kInputValues, kLastSynapticWeights), MatrixException);
	ASSERT_THROW(bp.CalculateDeltaWeights(kLearningRate, kMomentum, kLocalGradient, {}, kLastSynapticWeights), MatrixException);
	ASSERT_THROW(bp.CalculateDeltaWeights(kLearningRate, kMomentum, kLocalGradient, kInputValues, {}), MatrixException);
}

TEST(Back_propagation_tests, t2_calculate_delta_biases)
{
	BackPropagation bp = aux::CreateBackPropagationObj();
	const double kLearningRate = 0.5;
	const double kMomentum = 0.5;
	const Vector<double> kLocalGradient{ 0.5,0.5 };
	const Vector<double> kLastDeltaBiases{ 0.1, 0.2 };

	auto delta_biases_actual = bp.CalculateDeltaBiases(kLearningRate, kMomentum, kLocalGradient, kLastDeltaBiases);
	std::vector<double> delta_biases_expect{ 0.3, 0.35 };

	ASSERT_TRUE(aux::CompareVector(delta_biases_actual.GetVector(), delta_biases_expect));

	//Error check
	ASSERT_THROW(bp.CalculateDeltaBiases(kLearningRate, kMomentum, {}, kLastDeltaBiases), MatrixException);
	ASSERT_THROW(bp.CalculateDeltaBiases(kLearningRate, kMomentum, kLocalGradient, {}), MatrixException);
}

int main(int argc, char* argv[])
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
