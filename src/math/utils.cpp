#include "math/utils.hpp"


namespace cpp_ml::math::utils {


auto logistic_function (const double z) -> double {

	double logistic_value = 1.0 / (1.0 + std::exp(-z));

	return logistic_value;

}


}
