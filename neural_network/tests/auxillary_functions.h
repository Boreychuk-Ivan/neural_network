// Auxillary methods
#pragma once
#include "pch.h"

namespace aux
{
    template <class T>
    bool CompareVector(const std::vector<T>& kV_0, const std::vector<T>& kV_1)
    {
      if (kV_0.size() != kV_1.size()) return 0;
      for (size_t it = 0; it < kV_0.size(); ++it)
      {
        if (kV_0.at(it) != kV_1.at(it)) return 0;
      }
      return 1;
    }

}
