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
    LossFunctions() = delete;
    virtual double CalculateError(const Matrix<double>& kActual, const Matrix<double>& kPredicted) = 0;
};


class MeanSquareError : public LossFunctions
{
    double CalculateError(const Matrix<double>& kActual, const Matrix<double>& kPredicted) final;
};

class RootMeanSquareError : public LossFunctions
{
    double CalculateError(const Matrix<double>& kActual, const Matrix<double>& kPredicted) final;
};

class LossFunctionsFabric
{
    std::shared_ptr<LossFunctions> CreateLossFunction(const LossFunctionType&);
};