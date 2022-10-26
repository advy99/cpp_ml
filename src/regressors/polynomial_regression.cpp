#include "regressors/polynomial_regression.hpp"

namespace cpp_ml::models::regressors {

polynomial_regression :: polynomial_regression ()
	:degree_(1)
{}

polynomial_regression :: polynomial_regression (const std::size_t degree)
	:degree_(degree)
{}

polynomial_regression :: ~polynomial_regression() {}

auto polynomial_regression :: predict(const std::vector<double> & instance) const -> double {

	double prediction = weights_[0];

	auto polynomial_instance = convert_instance_to_polynomial(instance);

	for (std::size_t i = 1; i < weights_.size(); ++i) {
		prediction += weights_[i] * polynomial_instance[i - 1];
	}

	return prediction;

}

auto polynomial_regression :: predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<double>  {
	std::vector<double> predictions;

	predictions.reserve(new_data.size());

	for (const auto & data : new_data ) {
		predictions.push_back(predict(data));
	}

	return predictions;

}

auto polynomial_regression :: fit (const std::vector<std::vector<double>> & data, const std::vector<double> & targets) -> void {

	// first, add the variables of the data but powered
	std::vector<std::vector<double>> polynomial_data;

	for (const auto & row : data) {
		polynomial_data.push_back(convert_instance_to_polynomial(row));
	}


	// add the dummy value for calculate the bias
	std::vector<std::vector<double>> data_with_dummie;

	for (auto & row : polynomial_data) {
		std::vector<double> new_row {1.0};
		new_row.insert(new_row.end(), row.begin(), row.end());
		data_with_dummie.push_back(new_row);
	}


	// least squared method with matrix
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

	// save the computed weights
	for (std::size_t i = 0; i < weights.size(); ++i) {
		weights_[i] = weights[i][0];
	}

}

auto polynomial_regression :: convert_instance_to_polynomial(
		const std::vector<double> & instance
) const -> std::vector<double> {

	std::vector<double> result = instance;

	// for every column
	for (const auto & variable : instance) {
		// we add the current column powered to the degree. Start at 2 because
		// pow(variable, 1) is the same variable
		for (std::size_t current_degree = 2; current_degree <= degree_; ++current_degree) {
			result.push_back(std::pow(variable, current_degree));
		}
	}

	return result;

}


} // end namespace models::regressors
