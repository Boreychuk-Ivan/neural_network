#pragma once

#include <vector>
#include <iostream>
#include <initializer_list>
#include <assert.h>
#include <cmath>  //fmod
#include <limits> //numeric_limits

#include "matrix_exception.h"

template <class T>
class Matrix
{
protected:
	size_t m_rows, m_cols;
	std::vector<T> m_matrix;

public:
    //Constructors
	Matrix() : m_rows(0), m_cols(0){};
	Matrix(const int kRows, const int kCols);
	Matrix(const int kRows, const int kCols, T *mtx);
    Matrix(const int kRows, const int kCols, const std::vector<T>& kVector);
	Matrix(const std::initializer_list<T> &list);

	Matrix(const Matrix &mtx);
	Matrix &operator=(const Matrix kRightMtx);
	Matrix &operator=(const std::initializer_list<T> &list);
	Matrix &operator=(const T *);

    //Element callers
	T& at(size_t cell);
	T& at(size_t row, size_t col);

	T at(size_t cell) const;
	T at(size_t row, size_t col) const;

    //Getters
	std::vector<T> GetVector() const;
	Matrix GetRow(const size_t) const;
    Matrix GetCol(const size_t) const;
	size_t GetSize() const { return m_rows * m_cols; }
	size_t GetRowsNum() const { return m_rows; }
	size_t GetColsNum() const { return m_cols; }
    Matrix<T> GetMtx(const size_t kRowBeg, const size_t kColBeg, const size_t kRowEnd, const size_t kColEnd);

    //Setters
    void SetRow(const size_t kRow, const std::vector<T>& kData);
    void SetCol(const size_t kCol, const std::vector<T>& kData);

    //Methods
	bool IsEqualSize(const Matrix kOther) const;
    T SumElements() const;
    T MaxElement() const;
    T MinElement() const;

    //Arithmetic
    Matrix<T> DotMult(const Matrix<T> &kRightMtx) const;
    
    //Creators
    void InitialiseDiag(const size_t& kSize);

    //Operators
	template <class Tf>
	friend Matrix<Tf> operator+(const Matrix<Tf> &kLeftMtx, const Matrix<Tf> &kRightMtx);

	template <class Tf>
	friend Matrix<Tf> operator+(const Matrix<Tf> &kLeftMtx, Tf num);

	template <class Tf>
	friend Matrix<Tf> operator+(Tf num, const Matrix<Tf> &kRightMtx);

	template <class Tf>
	friend Matrix<Tf> operator-(const Matrix<Tf> &kLeftMtx, const Matrix<Tf> &kRightMtx);

	template <class Tf>
	friend Matrix<Tf> operator-(const Matrix<Tf> &kLeftMtx, Tf num);

	template <class Tf>
	friend Matrix<Tf> operator*(const Matrix<Tf> &kLeftMtx, const Matrix<Tf> &kRightMtx);

	friend Matrix<bool> operator*(const Matrix<bool> &kLeftMtx, const Matrix<bool> &kRightMtx);

	template <class Tf>
	friend Matrix<Tf> operator*(const Matrix<Tf> &kLeftMtx, const Tf &num);

	template <class Tf>
	friend Matrix<Tf> operator*(const Tf &num, const Matrix<Tf> &kRightMtx);

	template <class Tf>
	friend Matrix<Tf> operator!(const Matrix<Tf> &kRightMtx); //Transpose

	template <class Tf>
	friend bool operator==(const Matrix<Tf>& kLeftMtx, const Matrix<Tf>& kRightMtx);

	template <class Tf>
	friend std::ostream &operator<<(std::ostream &out, const Matrix<Tf> &kRightMtx);
};

template <class T>
Matrix<T>::Matrix(const int kRows, const int kCols) : m_rows(kRows), m_cols(kCols)
{
	MatrixException::assert_throw(kRows > 0 && kCols > 0, "Error <Matrix> : invalid contructor parametrs\n");
	m_matrix = std::vector<T>(static_cast<int64_t>(kRows) * static_cast<int64_t>(kCols));
}

template <class T>
Matrix<T>::Matrix(const int kRows, const int kCols, T* mtx) : Matrix(kRows, kCols)
{
	for (size_t it = 0; it < m_rows * m_cols; ++it)
		m_matrix.at(it) = (*(mtx + it));
}

template<class T>
Matrix<T>::Matrix(const int kRows, const int kCols, const std::vector<T>& kVector) : Matrix(kRows, kCols)
{
    MatrixException::assert_throw(kVector.size() == kRows * kCols, 
		"Error <Matrix> : invalid contructor parametrs\n");
    m_matrix = kVector;
}

template <class T>
Matrix<T>::Matrix(const std::initializer_list<T> &list)
{
	m_cols = static_cast<size_t>(list.size());
	m_rows = 1;
	m_matrix.reserve(m_rows * m_cols);
	for (auto &num : list)
		m_matrix.push_back(num);
}

template <class T>
Matrix<T>::Matrix(const Matrix &mtx)
{
	m_rows = mtx.m_rows;
	m_cols = mtx.m_cols;
	m_matrix = mtx.m_matrix;
}

template <class T>
Matrix<T> &Matrix<T>::operator=(const Matrix kRightMtx)
{
	this->m_cols = kRightMtx.m_cols;
	this->m_rows = kRightMtx.m_rows;
	this->m_matrix = kRightMtx.m_matrix;
	return *this;
}

template <class T>
Matrix<T> &Matrix<T>::operator=(const std::initializer_list<T> &list)
{
	assert(list.size() == static_cast<size_t>(m_rows * m_cols));
	m_matrix.clear();
	for (auto &num : list)
	{
		m_matrix.push_back(num);
	}
	return *this;
}

template <class T>
Matrix<T> &Matrix<T>::operator=(const T *input_arr)
{
	m_matrix.clear();
	m_matrix.reserve(m_cols * m_rows);
	for (int it = 0; it < m_cols * m_rows; ++it)
	{
		m_matrix.push_back(*(input_arr + it));
	}
	return *this;
}

template<class T>
inline T& Matrix<T>::at(size_t cell)
{
	MatrixException::assert_throw(cell < m_rows * m_cols, "Error <at> : invalid cell number\n");	
	return m_matrix.at(cell);
}

template<class T>
inline T& Matrix<T>::at(size_t row, size_t col)
{
	MatrixException::assert_throw(row <= m_rows && col <= m_cols, "Error <at> : invalid input parametrs\n");
	return m_matrix.at(row * m_cols + col);
}

template<class T>
inline T Matrix<T>::at(size_t cell) const
{
	MatrixException::assert_throw(cell < m_rows * m_cols, "Error <at> : invalid cell number\n");
	return m_matrix.at(cell);
}

template<class T>
inline T Matrix<T>::at(size_t row, size_t col) const
{
	MatrixException::assert_throw(row <= m_rows && col <= m_cols, "Error <at> : invalid input parametrs\n");
	return m_matrix.at(row * m_cols + col);
}

template <class T>
Matrix<T> Matrix<T>::GetRow(const size_t kRow) const
{
	MatrixException::assert_throw(kRow <= m_rows, 
		"Error <GetRow> : Invalid row number\n");
	Matrix<T> row_vector(1, (int)m_cols);
	for (int col = 0; col < m_cols; ++col)
		row_vector.at(col) = (this->at(kRow, col));
	return row_vector;
}

template <class T>
Matrix<T> Matrix<T>::GetCol(const size_t kCol) const
{
	MatrixException::assert_throw(kCol <= m_cols, "Error <GetCol> : Invalid col number\n");
	Matrix<T> col_vector((int)m_rows, 1);
	for (int row = 0; row < m_rows; ++row)
		col_vector.at(row) = (this->at(row, kCol));
	return col_vector;
}

template<class T>
Matrix<T> Matrix<T>::GetMtx(const size_t kRowBeg, const size_t kColBeg, const size_t kRowEnd, const size_t kColEnd)
{
	MatrixException::assert_throw((kRowBeg <= kRowEnd) || (kColBeg <= kColEnd), "Error <GetMtx> : invalid matrix parametrs\n");
    Matrix<T> part_mtx((int)(kRowEnd - kRowBeg + 1), (int)(kColEnd - kColBeg + 1));
    for(size_t row_it = 0; row_it <= kRowEnd - kRowBeg; ++row_it)
    {
        for(size_t col_it = 0; col_it <= kColEnd - kColBeg; ++col_it)
        {
            part_mtx.at(row_it, col_it) = this->at(row_it+ kRowBeg, col_it + kColBeg);
        }
    }
    return part_mtx;
}

template<class T>
void Matrix<T>::SetRow(const size_t kRow, const std::vector<T>& kData)
{
	MatrixException::assert_throw(kData.size() == m_cols && kRow <= m_rows, "Error <SetRow> : Invalid size of vector");
    for (int it = 0; it < m_cols; ++it)
        this->at(kRow, it) = kData.at(it);
}

template<class T>
void Matrix<T>::SetCol(const size_t kCol, const std::vector<T>& kData)
{
	MatrixException::assert_throw(kData.size() == m_rows && kCol <= m_cols, "Error <SetCol>: Invalid size of vector");
    for (int it = 0; it < m_rows; ++it)
        this->at(it, kCol) = kData.at(it);
}

template <class T>
inline bool Matrix<T>::IsEqualSize(const Matrix kOther) const
{
	if (m_cols == kOther.m_cols && m_rows == kOther.m_rows)
		return true;
	else
		return false;
}

template<class T>
inline T Matrix<T>::SumElements() const
{
    T accumulator = 0;
    for (int it = 0; it < m_cols*m_rows; ++it)
        accumulator += m_matrix.at(it);
    return accumulator;
}

template<class T>
T Matrix<T>::MaxElement() const
{
    T max_element = std::numeric_limits<T>::min();
    for (int it = 0; it < m_matrix.size(); ++it)
    {
        if (max_element < m_matrix.at(it))
        {
            max_element = m_matrix.at(it);
        }
    }
    return max_element;
}

template<class T>
T Matrix<T>::MinElement() const
{
    T min_element = std::numeric_limits<T>::max();
    for (int it = 0; it < m_matrix.size(); ++it)
    {
        if (min_element > m_matrix.at(it))
        {
            min_element = m_matrix.at(it);
        }
    }
    return min_element;
}

template <class T>
inline Matrix<T> Matrix<T>::DotMult(const Matrix<T> &kRightMtx) const
{
	MatrixException::assert_throw(m_cols == kRightMtx.m_cols && m_rows == kRightMtx.m_rows,
		"Error <DotMult> : Invalid matrix sizes\n");
    Matrix<T> result_mtx(*this);
	for (int it = 0; it < m_matrix.size(); ++it)
		result_mtx.at(it) *= kRightMtx.at(it);
	return result_mtx;
}

template<class T>
void Matrix<T>::InitialiseDiag(const size_t & kSize)
{
    Matrix<T> diag_mtx(kSize, kSize);
    for(int it = 0; it < kSize; ++it)
        diag_mtx.at(it*kSize+it) = 1;
    *this = diag_mtx;
}

template <class T>
std::vector<T> Matrix<T>::GetVector() const
{
	return m_matrix;
}

template <class T>
Matrix<T> operator+(const Matrix<T> &kLeftMtx, const Matrix<T> &kRightMtx)
{
	Matrix<T> mtx_sum(kLeftMtx);
	MatrixException::assert_throw(kLeftMtx.m_cols == kRightMtx.m_cols && 
		kLeftMtx.m_rows == kRightMtx.m_rows, "Error <operator+> : Invalid matrix sizes\n");

	for (int it(0); it < kLeftMtx.GetSize(); ++it)
	{
		mtx_sum.at(it) += kRightMtx.at(it);
	}
	return mtx_sum;
}

template <class Tf>
Matrix<Tf> operator+(const Matrix<Tf> &kLeftMtx, Tf num)
{
	Matrix<Tf> mtx_sum(kLeftMtx);
	for (int it(0); it < kLeftMtx.GetSize(); ++it)
		mtx_sum.at(it) += num;
	return mtx_sum;
}

template <class Tf>
Matrix<Tf> operator+(Tf num, const Matrix<Tf> &kRightMtx) { return (kRightMtx + num); }

template <class Tf>
Matrix<Tf> operator-(const Matrix<Tf> &kLeftMtx, Tf num)
{
	Matrix<Tf> mtx_diff(kLeftMtx);
	for (int it(0); it < kLeftMtx.GetSize(); ++it)
		mtx_diff.at(it) = kLeftMtx.at(it) - num;
	return mtx_diff;
}

template <class Tf>
Matrix<Tf> operator-(const Matrix<Tf> &kLeftMtx, const Matrix<Tf> &kRightMtx)
{
	Matrix<Tf> mtx_diff(kLeftMtx);
	MatrixException::assert_throw(kLeftMtx.m_cols == kRightMtx.m_cols && 
		kLeftMtx.m_rows == kRightMtx.m_rows, "Error <operator-> : Invalid matrix sizes\n");

	for (int it(0); it < kLeftMtx.GetSize(); ++it)
		mtx_diff.at(it) = kLeftMtx.at(it) - kRightMtx.at(it);
	return mtx_diff;
}

template <class Tf>
Matrix<Tf> operator*(const Matrix<Tf> &kLeftMtx, const Matrix<Tf> &kRightMtx)
{
	MatrixException::assert_throw(kLeftMtx.m_cols == kRightMtx.m_rows, 
		"Error <operator*> : Invalid matrix sizes\n");
	Matrix<Tf> mtx_prod((int)kLeftMtx.m_rows, (int)kRightMtx.m_cols);

	for (int row = 0; row < kLeftMtx.m_rows; ++row)
	{
		for (int col = 0; col < kRightMtx.m_cols; ++col)
		{
			std::vector<Tf> a_row = kLeftMtx.GetRow(row).m_matrix;
			std::vector<Tf> b_col = kRightMtx.GetCol(col).m_matrix;

			std::vector<Tf> product_elements;
			for (int it = 0; it < a_row.size(); ++it)
			{
				product_elements.push_back(a_row.at(it) * b_col.at(it));
			}

			Tf product_cell = 0;
			for (auto &num : product_elements)
			{
				product_cell += num;
			}

			mtx_prod.at(row, col) = product_cell;
		}
	}
	return mtx_prod;
}

template <class Tf>
Matrix<Tf> operator*(const Matrix<Tf> &kLeftMtx, const Tf &num)
{
	Matrix<Tf> product_mtx(kLeftMtx);
	for (int it = 0; it < kLeftMtx.GetSize(); ++it)
		product_mtx.at(it) = kLeftMtx.at(it) * num;
	return product_mtx;
}

template <class Tf>
Matrix<Tf> operator*(const Tf &num, const Matrix<Tf> &kRightMtx)
{ 
	return (kRightMtx * num); 
}

template <class Tf>
Matrix<Tf> operator%(const Matrix<Tf> &kLeftMtx, const Tf &num)
{
	Matrix<Tf> result_mtx(kLeftMtx);
	for (int it = 0; it < kLeftMtx.GetSize(); ++it)
	{
		result_mtx.at(it) = (kLeftMtx.at(it) % num);
	}
	return result_mtx;
}

template <>
inline Matrix<double> operator%(const Matrix<double> &kLeftMtx, const double &num)
{
	Matrix<double> result_mtx(kLeftMtx);
	for (int it = 0; it < kLeftMtx.GetSize(); ++it)
	{
		result_mtx.at(it) = fmod(kLeftMtx.at(it), num);
	}
	return result_mtx;
}

template <>
inline Matrix<float> operator%(const Matrix<float> &kLeftMtx, const float &num)
{
	Matrix<float> result_mtx(kLeftMtx);
	for (int it = 0; it < kLeftMtx.GetSize(); ++it)
	{
		result_mtx.at(it) = fmod(kLeftMtx.at(it), num);
	}
	return result_mtx;
}

template <class Tf>
Matrix<Tf> operator!(const Matrix<Tf> &kRightMtx)
{
	Matrix<Tf> transope_mtx((int)kRightMtx.m_cols, (int)kRightMtx.m_rows);
	for (int row = 0; row < kRightMtx.m_rows; ++row)
	{
		for (int col = 0; col < kRightMtx.m_cols; ++col)
		{
			transope_mtx.at(col, row) = kRightMtx.at(row, col);
		}
	}
	return transope_mtx;
}

template<class Tf>
inline bool operator==(const Matrix<Tf>& kLeftMtx, const Matrix<Tf>& kRightMtx)
{
	if (!kLeftMtx.IsEqualSize(kRightMtx)) return 0;
	for (size_t it = 0; it < kLeftMtx.GetSize(); ++it)
		if (kLeftMtx.at(it) != kRightMtx.at(it)) return 0;
	return 1;

}

template <class T>
std::ostream &operator<<(std::ostream &out, const Matrix<T> &kRightMtx)
{
	out << std::endl;
	for (int row(0); row < kRightMtx.GetRowsNum(); ++row)
	{
		for (int col(0); col < kRightMtx.GetColsNum(); ++col)
		{
			out << (kRightMtx.at(row, col) >= 0 ? " " : "") << kRightMtx.at(row, col) << "\t";
		}
		out << std::endl;
	}
	return out;
}

template <>
inline std::ostream &operator<<(std::ostream &out, const Matrix<uint8_t> &kRightMtx)
{
	out << std::endl;
	for (int row(0); row < kRightMtx.GetRowsNum(); ++row)
	{
		for (int col(0); col < kRightMtx.GetColsNum(); ++col)
		{
			out << (kRightMtx.at(row, col) >= 0 ? " " : "") << static_cast<unsigned int>(kRightMtx.at(row, col)) << "\t";
		}
		out << std::endl;
	}
	return out;
}


//Class Vector
namespace VTYPE
{
enum VTYPE_ENUM
{
	ROW,
	COL
};
}

template <class T, VTYPE::VTYPE_ENUM VECTOR_TYPE = VTYPE::ROW>
class Vector : public Matrix<T>
{
private:
	using Matrix::m_cols;
	using Matrix::m_matrix;
	using Matrix::m_rows;

public:
	Vector() : Matrix<T>(){};
	Vector(const std::initializer_list<T> &list) : Matrix(list)
	{
		m_rows = VECTOR_TYPE == VTYPE::ROW ? 1 : list.size();
		m_cols = VECTOR_TYPE == VTYPE::ROW ? list.size() : 1;
	};
	Vector(const size_t size)
	{
		m_matrix.reserve(size);
		m_matrix.insert(m_matrix.begin(), size, 0);
		m_rows = VECTOR_TYPE == VTYPE::ROW ? 1 : size;
		m_cols = VECTOR_TYPE == VTYPE::ROW ? size : 1;
	};
	Vector(const size_t size, T *vector) : Matrix<T>(1, size)
	{
		m_matrix.reserve(size);
		m_matrix.insert(m_matrix.begin(), vector, vector + size + 1);
		m_rows = VECTOR_TYPE == VTYPE::ROW ? 1 : size;
		m_cols = VECTOR_TYPE == VTYPE::ROW ? size : 1;
	};

	Vector(const Matrix<T> &kMtx)
	{
		assert(kMtx.GetColsNum() == 1 || kMtx.GetRowsNum() == 1);
		m_cols = kMtx.GetColsNum();
		m_rows = kMtx.GetRowsNum();
		m_matrix = kMtx.GetVector();
	}

	bool IsRow() const { return (VECTOR_TYPE == VTYPE::ROW ? 1 : 0); }
	bool IsCol() const { return (VECTOR_TYPE == VTYPE::COL ? 1 : 0); }

	using Matrix::operator=;
};
