#include <fstream>
#include <sstream>
#include <vector>

namespace cpp_ml::datasets::reading {

template <typename T, typename U>
auto read_x_y_from_csv (
	const std::string & file_path,
	const bool has_header,
	const char comment_char,
	const char separator
) -> std::pair<std::vector<std::vector<T>>, std::vector<U>> {

	std::vector<std::vector<T>> data;
	std::vector<U> targets;

	std::ifstream f_stream (file_path, std::ios::in);

	std::string line;

	std::getline(f_stream, line);

	if (has_header) {
		std::getline(f_stream, line);
	}


	while ( f_stream ) {

		if (line[0] != comment_char) {
			std::istringstream line_stream (line);

			std::vector<std::string> values;

			std::string value;

			std::getline(line_stream, value, separator);

			while (line_stream) {
				values.push_back(value);

				std::getline(line_stream, value, separator);
			}

			std::vector<T> t_values(values.size() - 1);

			for (std::size_t i = 0; i < values.size() - 1; ++i) {
				T val;

				if constexpr (std::is_floating_point<T>::value) {
					val = std::stod(values[i]);
				} else if constexpr (std::is_integral<T>::value) {
					val = std::stoi(values[i]);
				}

				t_values[i] = static_cast<T>(val);
			}

			U target = static_cast<U>(values.back());

			data.push_back(t_values);
			targets.push_back(target);

		}

		std::getline(f_stream, line);

	}

	return std::make_pair(data, targets);

}


} //end datasets::reading namespace
