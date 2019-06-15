#pragma once

#include <string>
#include <exception>

namespace err
{
	class NNException : public std::exception
	{
	private:
		std::string m_error;
	public:
		NNException(std::string error) : m_error(error) {}

		const char* what() const noexcept { return m_error.c_str(); }
	};
	void assert_throw(const bool kCondition, const std::string& kMessege);
}