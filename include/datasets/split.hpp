#ifndef SPLIT_HPP
#define SPLIT_HPP

#include <string>
#include <vector>

namespace cpp_ml::datasets::split {

template <typename T, typename U>
auto train_test_split (
	const std::vector<std::vector<T>> & data,
	const std::vector<U> & targets,
	const double test_size = 0.8,
	const uint64_t seed = 42
) -> std::tuple<std::vector<std::vector<T>>, std::vector<U>,
	  				 std::vector<std::vector<T>>, std::vector<U>>;

} //end datasets::reading namespace


#include "datasets/split.tpp"


#endif
