#pragma once

namespace yukilib {

using fp_t = double;

namespace numbers {

#if defined(__cpp_lib_math_constants) and __cpp_lib_math_constants

#include <numbers>
using namespace std::numbers;

#else
template<typename T> inline constexpr T pi_v = static_cast<T>(3.141592653589793238462643383279502);
//inline constexpr double pi = pi_v<double>;

#endif

}

}  // namespace yukilib

