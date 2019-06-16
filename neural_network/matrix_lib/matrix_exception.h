#pragma once

#include <string>
#include <exception>

class MatrixException : public std::exception
{
private:
	std::string m_error;
public:
	MatrixException(std::string error) : m_error(error) {}

	static void assert_throw(const bool kCondition, const std::string& kMessege)
	{
		if (!kCondition)
		{
			throw(MatrixException(kMessege));
		}
	}

	const char* what() const noexcept { return m_error.c_str(); }
};


