#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "regressors.hpp"
#include <functional>

namespace cpp_ml::models::regressors {

class linear_regression : public regressor {
	private:
		std::vector<double> weights_ {};

	public:

		virtual auto predict(const std::vector<double> & instance) const -> double override;
		virtual auto predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<double> override;

		virtual auto fit(const std::vector<std::vector<double> > & data, const std::vector<double> & targets) -> void override;

};


} // end of models::regressors namespace

#endif
