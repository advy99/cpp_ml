#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "regressors.hpp"
#include <functional>

namespace cpp_ml::models::regressors {

class linear_regression : public polynomial_regression {

	public:
	linear_regression();
};


} // end of models::regressors namespace

#endif
