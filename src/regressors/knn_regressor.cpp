#include "regressors/knn_regressor.hpp"
#include "math/distances.hpp"

#include <vector>
#include <map>

namespace cpp_ml::models::regressors {

knn_regressor :: knn_regressor(int32_t k, const std::function<double(const std::vector<double> &, const std::vector<double> &)> & distance_f) {
	k_ = k;
	distance_function_ = distance_f;
}

auto knn_regressor :: fit(const std::vector<std::vector<double> > & data, const std::vector<double> & targets) -> void {
	data_ = data;
	targets_ = targets;
}

auto knn_regressor :: predict(const std::vector<double> & instance) const -> double {

	// compute the k closer's neighbors
	auto neighbors = math::distances::compute_k_nearest_neighbors(instance, data_, k_, distance_function_);

	// compute the prediction as the mean of the neighbors values
	double prediction = 0.0;

	for (const auto & index : neighbors) {
		prediction += targets_[index];
	}

	prediction = prediction / static_cast<double>(neighbors.size());

	return prediction;


}

auto knn_regressor :: predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<double>  {
	std::vector<double> predictions;

	predictions.reserve(new_data.size());

	for (const auto & data : new_data ) {
		predictions.push_back(predict(data));
	}

	return predictions;

}



} // ends models::regressors namespace
