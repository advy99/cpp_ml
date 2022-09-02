#ifndef REGRESSION_METRICS
#define REGRESSION_METRICS

#include <vector>
#include <cstdint>

namespace cpp_ml::metrics::regression_metrics {

auto mean_absolute_error (const std::vector<double> & y_true, const std::vector<double> & y_pred) -> double;

auto mean_squared_error (const std::vector<double> & y_true, const std::vector<double> & y_pred) -> double;

auto root_mean_squared_error (const std::vector<double> & y_true, const std::vector<double> & y_pred) -> double;


} // end metrics::regression_metrics namespace



#endif
