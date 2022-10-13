#include <math/distances.hpp>

#include <cmath>
#include <queue>
#include <functional>

namespace cpp_ml::math::distances {

auto minkowski_distance(const std::vector<double> & point_a, const std::vector<double> & point_b, int32_t p) -> double {
	// minkowski distance as in the formula
	double distance = 0.0;

	for (std::size_t i = 0; i < point_a.size(); i++) {
		double abs_difference = std::abs(point_a[i] - point_b[i]);
		distance += std::pow(abs_difference, p);
	}

	distance = std::pow(distance, 1.0 / p);

	return distance;
}

auto euclidean_distance(const std::vector<double> & point_a, const std::vector<double> & point_b) -> double {
	// euclidean distance is the minkowski distance with p = 2
	return minkowski_distance(point_a, point_b, 2);
}

auto manhattan_distance(const std::vector<double> & point_a, const std::vector<double> & point_b) -> double {
	// euclidean distance is the minkowski distance with p = 1
	return minkowski_distance(point_a, point_b, 1);
}



auto compute_k_nearest_neighbors(const std::vector<double> & instance,
											const std::vector<std::vector<double>> & data,
											const size_t k,
											std::function<double(const std::vector<double> &, const std::vector<double> &)> distance_function
) -> std::vector<std::size_t> {

	// a vector for the neighbors and a priority_queue with the distances
	std::vector<std::size_t> neighbors;
	// we use std::greater, so closest distances are in top.
	std::priority_queue<std::pair<double, std::size_t>,
							  std::vector<std::pair<double,
							  std::size_t>>, std::greater<std::pair<double, std::size_t>>> distances;

	// compute the distance of the instance with every data point
	for (std::size_t i = 0; i < data.size(); i++ ) {
		distances.push(std::make_pair(distance_function(instance, data[i]), i) );
	}

	neighbors.resize(k);

	// select the k closer's neighbors
	for (std::size_t i = 0; i < neighbors.size(); i++ ) {
		neighbors[i] = distances.top().second;
		distances.pop();
	}

	return neighbors;

}


}
