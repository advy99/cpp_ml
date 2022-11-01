#include "regressors/linear_regression.hpp"
#include "math/matrix.hpp"

namespace cpp_ml::models::regressors {

linear_regression :: linear_regression ()
	:polynomial_regression(1)
{}

linear_regression :: ~linear_regression() {}

} // end namespace models::regressors
