#include "metrics/classification_metrics.hpp"


namespace cpp_ml::metrics::classification_metrics {

auto accuracy_score (const std::vector<int32_t> & y_true, const std::vector<int32_t> & y_pred) -> double {

	uint32_t num_corrects = 0;

	// count corrects predictions
	for (std::size_t i = 0; i < y_true.size(); ++i) {
		if (y_true[i] == y_pred[i]) {
			num_corrects += 1;
		}
	}

	// divide by total number of instances
	return (static_cast<double>(num_corrects) / static_cast<double>(y_true.size()));

}



} // end metrics::classification_metrics namespace
