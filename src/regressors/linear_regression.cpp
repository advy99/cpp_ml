#include "regressors/linear_regression.hpp"
#include "math/matrix.hpp"

namespace cpp_ml::models::regressors {


auto linear_regression :: predict(const std::vector<double> & instance) const -> double {

	double prediction = weights_[0];

	for (std::size_t i = 1; i < weights_.size(); ++i) {
		prediction += weights_[i] * instance[i - 1];
	}

	return prediction;

}

auto linear_regression :: predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<double>  {
	std::vector<double> predictions;

	predictions.reserve(new_data.size());

	for (const auto & data : new_data ) {
		predictions.push_back(predict(data));
	}

	return predictions;

}

auto linear_regression :: fit (const std::vector<std::vector<double>> & data, const std::vector<double> & targets) -> void {

	std::vector<std::vector<double>> data_with_dummie;

	for (auto & row : data) {
		std::vector<double> new_row {1.0};
		new_row.insert(new_row.end(), row.begin(), row.end());
		data_with_dummie.push_back(new_row);
	}

	
	auto x_transposed = math::matrix::transpose(data_with_dummie);


	auto pseudoinverse = math::matrix::dot(x_transposed, data_with_dummie);
	pseudoinverse = math::matrix::invert(pseudoinverse);
	pseudoinverse = math::matrix::dot(pseudoinverse, x_transposed);

	std::vector<std::vector<double>> targets_as_column;
	targets_as_column.push_back(targets);
	targets_as_column = math::matrix::transpose(targets_as_column);

	auto weights = math::matrix::dot(pseudoinverse, targets_as_column);

	weights_.clear();
	weights_.resize(weights.size());

	for (std::size_t i = 0; i < weights.size(); ++i) {
		weights_[i] = weights[i][0];
	}

}



} // end namespace models::regressors
