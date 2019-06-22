#pragma once

#include "pch.h"
#include "matrix_lib/matrix_lib.h"
#include "auxillary_functions.h"

#include <gtest/gtest.h>



// Tests
TEST(Matrix_tests, t0_constructors)
{
	ASSERT_THROW(Matrix<int> mtx(-1, -1), MatrixException);

	Matrix<int> matrix_0(2, 2);
	for (size_t it = 0; it < matrix_0.GetSize(); ++it)
		ASSERT_TRUE(matrix_0.at(it) == 0);

	Matrix<int> matrix_1{0,1,2,3,4,5,6,7,8,9};
	ASSERT_TRUE(matrix_1.GetRowsNum() == 1 && matrix_1.GetColsNum() == 10);

	Matrix<int> matrix_2{ 2, 5, {0,1,2,3,4,5,6,7,8,9} };
	ASSERT_TRUE(matrix_2.GetRowsNum() == 2 && matrix_2.GetColsNum() == 5);
	ASSERT_TRUE(aux::CompareVector<int>(matrix_2.GetRow(0).GetVector(), { 0,1,2,3,4 }));
	ASSERT_TRUE(aux::CompareVector<int>(matrix_2.GetRow(1).GetVector(), { 5,6,7,8,9}));
}

TEST(Matrix_tests, t1_get_matrix_part)
{
	Matrix<int> matrix(4, 4, { 0,1,2,3,  4,5,6,7,  8,9,10,11,  12,13,14,15 });
	Matrix<int> matrix_part = matrix.GetMtx(0,0,1,1);
	Matrix<int> matrix_part_expect(2, 2, { 0,1, 4,5 });
	ASSERT_TRUE(matrix_part == matrix_part_expect);
}

TEST(Matrix_tests, t2_is_equal_size)
{
	Matrix<int> matrix_0(4, 5);
	Matrix<int> matrix_1(4, 5);
	Matrix<int> matrix_2(5, 4);
	ASSERT_TRUE(matrix_0.IsEqualSize(matrix_1));
	ASSERT_FALSE(matrix_0.IsEqualSize(matrix_2));
}

TEST(Matrix_tests, t3_sum_elements)
{
	Matrix<int> matrix_0(1, 5, {0,1,2,3,4});
	int expect_sum = 0 + 1 + 2 + 3 + 4;
	ASSERT_EQ(matrix_0.SumElements(), expect_sum);
}

TEST(Matrix_tests, t4_max_min_elements)
{
	Matrix<int> matrix_0(1, 5, { -2,-1,0,1,2 });
	int expect_min = -2;
	int expect_max = 2;
	ASSERT_EQ(matrix_0.MinElement(), expect_min);
	ASSERT_EQ(matrix_0.MaxElement(), expect_max);
}

TEST(Matrix_tests, t5_operator_plus_minus_mtx_mtx)
{
	Matrix<int> matrix_0(2, 2, { 1,2,3,4 });
	Matrix<int> matrix_1(2, 2, { 4,3,2,1 });
	Matrix<int> matrix_sum  = matrix_0 + matrix_1;
	Matrix<int> matrix_diff = matrix_0 - matrix_1;

	Matrix<int> expect_sum (2, 2, { 5,5,5,5 });
	Matrix<int> expect_diff(2, 2, {-3,-1,1,3});

	ASSERT_TRUE(matrix_sum == expect_sum);
	ASSERT_TRUE(matrix_diff == expect_diff);
}

TEST(Matrix_tests, t6_operator_plus_minus_mtx_num)
{
	Matrix<int> matrix_0(2, 2, { 1,2,3,4 });
	Matrix<int> matrix_sum = matrix_0 + 5;
	Matrix<int> matrix_diff = matrix_0 - 4;

	Matrix<int> expect_sum(2, 2, { 6,7,8,9 });
	Matrix<int> expect_diff(2, 2, { -3,-2,-1,0 });

	ASSERT_TRUE(matrix_sum == expect_sum);
	ASSERT_TRUE(matrix_diff == expect_diff);
}

TEST(Matrix_tests, t7_operator_mult_mtx)
{
	Matrix<int> matrix_0(2, 2, { 1,2,3,4 });
	Matrix<int> matrix_mult_0 = matrix_0 * matrix_0;
	Matrix<int> matrix_mult_1 = matrix_0 * 5;

	Matrix<int> expect_mult_0(2, 2, { 7,10, 15,22});
	Matrix<int> expect_mult_1(2, 2, { 5,10, 15,20});

	ASSERT_TRUE(matrix_mult_0 == expect_mult_0);
	ASSERT_TRUE(matrix_mult_1 == expect_mult_1);
}

TEST(Matrix_tests, t8_transpose)
{
	Matrix<int> matrix_0(2, 2, { 1,2,3,4 });
	Matrix<int> matrix_transpose = !matrix_0;
	Matrix<int> expect_transpose(2, 2, {1,3, 2,4});
	ASSERT_TRUE(matrix_transpose == expect_transpose);
}