#ifndef DISTANCES_HPP
#define DISTANCES_HPP

#include <vector>
#include <cstdint>
#include <functional>

namespace cpp_ml::math::distances {

auto minkowski_distance(const std::vector<double> & point_a, const std::vector<double> & point_b, int32_t p) -> double;

auto euclidean_distance(const std::vector<double> & point_a, const std::vector<double> & point_b) -> double;

auto manhattan_distance(const std::vector<double> & point_a, const std::vector<double> & point_b) -> double;

auto compute_k_nearest_neighbors(const std::vector<double> & instance,
											const std::vector<std::vector<double>> & data,
											const int32_t k,
											std::function<double(const std::vector<double> &, const std::vector<double> &)> distance_function
) -> std::vector<std::size_t>;

} // end of math::distance namespace

#endif
