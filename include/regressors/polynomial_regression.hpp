#ifndef POLYNOMIAL_REGRESSION_HPP
#define POLYNOMIAL_REGRESSION_HPP

#include "regressors.hpp"
#include "math/matrix.hpp"
#include <functional>
#include <cmath>

namespace cpp_ml::models::regressors {

class polynomial_regression : public regressor {
	private:
		std::size_t degree_;
		std::vector<double> weights_ {};


		auto convert_instance_to_polynomial(const std::vector<double> & instance) const -> std::vector<double>;

	public:

		polynomial_regression();
		polynomial_regression(const std::size_t degree);

		virtual auto predict(const std::vector<double> & instance) const -> double override;
		virtual auto predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<double> override;

		virtual auto fit(const std::vector<std::vector<double> > & data, const std::vector<double> & targets) -> void override;

};


} // end of models::regressors namespace

#endif
