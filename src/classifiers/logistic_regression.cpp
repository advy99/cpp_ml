#include "classifiers/logistic_regression.hpp"
#include "math/utils.hpp"
#include "random.hpp"

namespace cpp_ml::models::classifiers {

logistic_regression :: logistic_regression (const double learning_rate, const int32_t max_iters, const uint64_t seed)
	:learning_rate_ (learning_rate), max_iters_ (max_iters), random_seed_ (seed)
{}

logistic_regression :: ~logistic_regression() {}

auto logistic_regression :: predict(const std::vector<double> & instance) const -> int32_t {
	// TODO: Modify logistic regression to multiclass problems, not only binary problems ( One VS All ? )

	int32_t prediction = 0;

	double logistic_value = compute_logistic_function(instance);

	if (logistic_value >= 0.5) {
		prediction = 1;
	}

	return prediction;
}


auto logistic_regression :: predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<int32_t> {
	std::vector<int32_t> predictions;

	for (const auto & row : new_data) {
		predictions.push_back( predict(row) );
	}

	return predictions;
}

auto logistic_regression :: fit(const std::vector<std::vector<double> > & data, const std::vector<int32_t> & targets) -> void {

	weights_.clear();
	// a weight per column plus one for the bias
	weights_.resize(data[0].size() + 1);

	// start with random weights
	Random::set_seed(random_seed_);

	for (double & weight : weights_) {
		weight = Random::next_double(-50.0, 50.0);
	}

	for (int32_t i = 0; i < max_iters_; ++i) {

		for (std::size_t j = 0; j < data.size(); ++j) {

			double logistic_value = compute_logistic_function(data[j]);

			double prediction_difference = targets[j] - logistic_value;

			// compute the bias as if the column value is 1.0
			weights_[0] = weights_[0] + learning_rate_ * prediction_difference * logistic_value * (1 - logistic_value); // * 1.0; Bias weight
			for (std::size_t k = 1; k < weights_.size(); ++k) {
				double change = learning_rate_ * prediction_difference * logistic_value * (1 - logistic_value) * data[j][k - 1];

				weights_[k] = weights_[k] + change;
			}

		}

	}

}


auto logistic_regression :: compute_logistic_function(const std::vector<double> & instance) const -> double {
	// start with the bias (has no corresponding column in an instance)
	double regression_value = weights_[0];

	for (std::size_t i = 1; i < weights_.size(); ++i) {
		regression_value += weights_[i] * instance[i - 1];
	}

	double logistic_value = cpp_ml::math::utils::logistic_function(regression_value);

	return logistic_value;

}


}	// end cpp_ml::models::classifiers namespace
