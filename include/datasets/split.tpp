#include "datasets/split.hpp"

#include <tuple>
#include <random>
#include <numeric>
#include <algorithm>

namespace cpp_ml::datasets::split {

template <typename T, typename U>
auto train_test_split (
	const std::vector<std::vector<T>> & data,
	const std::vector<U> & targets,
	const double test_size,
	const uint64_t seed
) -> std::tuple<std::vector<std::vector<T>>, std::vector<U>,
	  				 std::vector<std::vector<T>>, std::vector<U>> {

	std::vector<std::vector<T>> x_train;
	std::vector<std::vector<T>> x_test;

	std::vector<U> y_train;
	std::vector<U> y_test;

	std::vector<std::size_t> index (data.size());
	std::iota(index.begin(), index.end(), 0);

	std::mt19937 random_generator;
	random_generator.seed(seed);

	std::shuffle(index.begin(), index.end(), random_generator);

	std::size_t i = 0;

	while (i < data.size() * test_size) {
		x_test.push_back(data[index[i]]);
		y_test.push_back(targets[index[i]]);
		++i;
	}

	while (i < data.size()) {
		x_train.push_back(data[index[i]]);
		y_train.push_back(targets[index[i]]);
		++i;
	}

	return std::make_tuple(x_train, y_train, x_test, y_test);

}


} //end datasets::reading namespace
