#include "loss_functions.h"

std::shared_ptr<LossFunctions> LossFunctionsFabric::CreateLossFunction(const LossFunctionType& kType)
{
	if (kType == MEAN_SQUARE_ERROR)
	{
		return std::make_shared<MeanSquareError>();
	}
	else if (kType == ROOT_MEAN_SQUARE_ERROR)
	{
		return std::make_shared<RootMeanSquareError>();
	}
	else
	{
		throw err::NNException("Error <LossFunctionsFabric>: Invalid loss functions\n");
	}
}

double MeanSquareError::CalculateError(const Matrix<double>& kActual, const Matrix<double>& kPredicted)
{
	err::assert_throw(kActual.IsEqualSize(kPredicted), "Error <MeanSquareError>: Invalid size of actual or predicted matrix\n");

    const size_t kBatchNumbers = kActual.GetRowsNum();
    const size_t kSize = kActual.GetColsNum();
    double sum_square_error = 0;
    for (int it = 0; it < kBatchNumbers; ++it)
    {
        auto error_row = kActual.GetRow(it) - kPredicted.GetRow(it);
        error_row = error_row.DotMult(error_row);
        sum_square_error += error_row.SumElements()/ kSize;
    }
    return (sum_square_error / kBatchNumbers);
}

double RootMeanSquareError::CalculateError(const Matrix<double>& kActual, const Matrix<double>& kPredicted)
{
	err::assert_throw(kActual.IsEqualSize(kPredicted), "Error <RootMeanSquareError> Invalid size of actual or predicted matrix\n");
    
    const size_t kBatchNumbers = kActual.GetRowsNum();
    const size_t kSize = kActual.GetColsNum();
    double sum_root_square_error = 0;
    for (int it = 0; it < kBatchNumbers; ++it)
    {
        auto error_row = kActual.GetRow(it) - kPredicted.GetRow(it);
        error_row = error_row.DotMult(error_row);
        sum_root_square_error += sqrt(error_row.SumElements()/ kSize);
    }
    return (sum_root_square_error / kBatchNumbers);
}
