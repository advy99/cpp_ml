#include "metrics/regression_metrics.hpp"

#include <cmath>

namespace cpp_ml::metrics::regression_metrics {

auto mean_absolute_error (const std::vector<double> & y_true, const std::vector<double> & y_pred) -> double {
	double mae = 0.0;

	for (std::size_t i = 0; i < y_true.size(); ++i) {
		mae += std::abs(y_true[i] - y_pred[i]);
	}

	return (mae / static_cast<double>(y_true.size()) );

}

auto mean_squared_error (const std::vector<double> & y_true, const std::vector<double> & y_pred) -> double {
	double mse = 0.0;

	for (std::size_t i = 0; i < y_true.size(); ++i) {
		mse += std::pow(y_true[i] - y_pred[i], 2);
	}

	return (mse / static_cast<double>(y_true.size()) );


}

auto root_mean_squared_error (const std::vector<double> & y_true, const std::vector<double> & y_pred) -> double {
	return std::sqrt( mean_squared_error(y_true, y_pred) );
}


} // end metrics::regression_metrics namespace
