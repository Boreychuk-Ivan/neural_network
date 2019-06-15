#include "error_handle.h"

void err::assert_throw(const bool kCondition, const std::string& kMessege)
{
	if (!kCondition)
	{
		throw(NNException(kMessege));
	}
}
