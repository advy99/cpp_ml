#include "transformers/standard_scaler.hpp"
#include "math/utils.hpp"

#include <cmath>

namespace cpp_ml::models::transformers {

standard_scaler :: standard_scaler ()
	: mean_ {0.0}, standard_deviation_ {0.0}
{}

auto standard_scaler :: transform (const std::vector<std::vector<double>> & data) const -> std::vector<std::vector<double>> {

	// we could make a copy in the method argument, but to hold consistency
	// with the data_transformer class
	std::vector<std::vector<double>> result = data;

	for (auto & row : result) {
		for (std::size_t i = 0; i < row.size(); ++i) {
			row[i] = (row[i] - mean_[i]) / standard_deviation_[i];
		}
	}

	return result;

}


auto standard_scaler :: fit (const std::vector<std::vector<double>> & data) -> void {

	// a mean per num of columns, and start with 0.0
	mean_ = std::vector<double> (data[0].size(), 0.0);
	standard_deviation_ = std::vector<double> (data[0].size(), 0.0);

	// compute the mean of each column
	for (const auto & row : data) {
		for (std::size_t i = 0; i < row.size(); ++i) {
			mean_[i] += row[i];
		}
	}

	for (double & column_mean : mean_) {
		column_mean = column_mean / static_cast<double>(data.size());
	}



	// compute the standard deviation for each column, using the mean we alredy computed
	for (const auto & row : data) {
		for (std::size_t i = 0; i < row.size(); ++i) {
			standard_deviation_[i] += std::pow(row[i] - mean_[i], 2);
		}
	}



	for (double & column_std_dev : standard_deviation_) {
		column_std_dev = column_std_dev * (1.0 / static_cast<double>(data.size()) );
		column_std_dev = std::sqrt(column_std_dev);

		// if the data does not variate, avoid divisions by 0.0
		if ( cpp_ml::math::utils::are_equal(column_std_dev, 0.0) ) {
			column_std_dev = 1.0;
		}

	}


}


auto standard_scaler :: get_means() const -> std::vector<double> {
	return mean_;
}

auto standard_scaler :: get_standard_deviations() const -> std::vector<double> {
	return standard_deviation_;
}



} // ends models::transformers namespace
