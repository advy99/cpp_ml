#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <limits>
#include <type_traits>
#include <cmath>

namespace cpp_ml::math::utils {

template <typename T> concept is_floating_point = std::is_floating_point_v<T>;

template <typename T>
requires is_floating_point<T>
auto are_equal(const T & lhs,
					const T & rhs,
					const T epsilon = std::numeric_limits<T>::epsilon()
) -> bool {
	return (std::abs(lhs - rhs) < epsilon);
}


auto logistic_function (const double z) -> double;


} // end math::utils namespace


#endif
