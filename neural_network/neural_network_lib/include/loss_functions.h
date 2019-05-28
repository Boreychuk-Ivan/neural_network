#pragma once

#include "pch.h"
#include "matrix_lib/matrix_lib.h"

enum LossFunctionType
{
    MEAN_SQUARE_ERROR,
    ROOT_MEAN_SQUARE_ERROR
};

class LossFunctions
{
public:
    virtual double CalculateError(const Matrix<double>& kActual, const Matrix<double>& kPredicted) = 0;
};


class MeanSquareError : public LossFunctions
{
public:
    double CalculateError(const Matrix<double>& kActual, const Matrix<double>& kPredicted) final;
};

class RootMeanSquareError : public LossFunctions
{
public:
    double CalculateError(const Matrix<double>& kActual, const Matrix<double>& kPredicted) final;
};

class LossFunctionsFabric
{
public:
    static std::shared_ptr<LossFunctions> CreateLossFunction(const LossFunctionType&);
};