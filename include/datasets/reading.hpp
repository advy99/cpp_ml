#ifndef READING_HPP
#define READING_HPP

#include <string>
#include <vector>

namespace cpp_ml::datasets::reading {

template <typename T, typename U>
auto read_x_y_from_csv (
	const std::string & file_path,
	const bool has_header,
	const char comment_char,
	const char separator
) -> std::pair<std::vector<std::vector<T>>, std::vector<U>>;


} //end datasets::reading namespace

#include "datasets/reading.tpp"

#endif
