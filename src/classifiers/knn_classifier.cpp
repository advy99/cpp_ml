#include "classifiers/knn_classifier.hpp"
#include "math/distances.hpp"

#include <vector>
#include <map>
#include <set>

namespace cpp_ml::models::classifiers {

knn_classifier :: knn_classifier(size_t k, const std::function<double(const std::vector<double> &, const std::vector<double> &)> & distance_f)
:
	distance_function_ { distance_f },
	k_ { k }
{ }

knn_classifier :: ~knn_classifier() {}

auto knn_classifier :: fit(const std::vector<std::vector<double> > & data, const std::vector<int32_t> & targets) -> void {
	this->classifier::fit(data, targets);
	data_ = data;
	targets_ = targets;
}


auto knn_classifier :: predict_probabilities(const std::vector<double> & instance) const -> std::vector<double> {

	std::vector<double> probabilities (num_classes_, 0.0);

	int32_t predicted_class = predict(instance);
	probabilities[static_cast<std::size_t>(predicted_class)] = 1.0;

	return probabilities;
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

auto knn_classifier :: predict_probabilities(const std::vector<std::vector<double>> & new_data) const -> std::vector<std::vector<double>> {
	std::vector<std::vector<double>> predictions;

	predictions.reserve(new_data.size());

	for (const auto & data : new_data ) {
		predictions.push_back(predict_probabilities(data));
	}

	return predictions;


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
