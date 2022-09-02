#include "classifiers/knn_classifier.hpp"
#include "math/distances.hpp"

#include <vector>
#include <map>

namespace cpp_ml::models::classifiers {

knn_classifier :: knn_classifier(int32_t k, const std::function<double(const std::vector<double> &, const std::vector<double> &)> & distance_f) {
	k_ = k;
	distance_function_ = distance_f;
}

auto knn_classifier :: fit(const std::vector<std::vector<double> > & data, const std::vector<int32_t> & targets) -> void {
	data_ = data;
	targets_ = targets;
}

auto knn_classifier :: predict(const std::vector<double> & instance) const -> int32_t {

	// compute the k closer's neighbors
	auto neighbors = math::distances::compute_k_nearest_neighbors(instance, data_, k_, distance_function_);

	std::map<int32_t, int32_t> count;

	// count the class of the neighbors
	for (auto neighbor: neighbors) {
		if (count.contains(targets_[neighbor])) {
			count[targets_[neighbor]] = count[targets_[neighbor]] + 1;
		} else {
			count[targets_[neighbor]] = 1;
		}
	}

	// the predicted class is the class of the majority closer's points
	auto predicted_class = std::max_element(count.begin(), count.end(),
							    [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
							        return p1.second < p2.second; });


	return predicted_class->first;


}

auto knn_classifier :: predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<int32_t>  {
	std::vector<int32_t> predictions;

	predictions.reserve(new_data.size());

	for (const auto & data : new_data ) {
		predictions.push_back(predict(data));
	}

	return predictions;

}



} // ends models::classifiers namespace
