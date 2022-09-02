#ifndef CLASSIFICATION_METRICS
#define CLASSIFICATION_METRICS

#include <vector>
#include <cstdint>

namespace cpp_ml::metrics::classification_metrics {

auto accuracy_score (const std::vector<int32_t> & y_true, const std::vector<int32_t> & y_pred) -> double;

} // end metrics::classification_metrics namespace



#endif
