#include "loss_functions.h"

double MeanSquareError::CalculateError(const Matrix<double>& kActual, const Matrix<double>& kPredicted)
{
    if (!kActual.IsEqualSize(kPredicted))
    {
        std::cout << "MSRE : Error! Invalid size of actual or predicted matrix\n";
    }

    const size_t kBatchNumbers = kActual.GetRowsNum();
    double sum_square_error = 0;
    for (int it = 0; it < kBatchNumbers; ++it)
    {
        auto error_row = kActual.GetRow(it) - kPredicted.GetRow(it);
        error_row = error_row.DotMult(error_row);
        sum_square_error = error_row.SumElements();
    }
    return (sum_square_error / kBatchNumbers);
}
