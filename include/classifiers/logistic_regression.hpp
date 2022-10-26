#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include "classifiers.hpp"
#include <functional>

namespace cpp_ml::models::classifiers {

class logistic_regression : public classifier {
	private:
		std::vector<double> weights_ {};
		double learning_rate_;
		int32_t max_iters_;
		const uint64_t random_seed_;


		auto compute_logistic_function(const std::vector<double> & instance) const -> double;

	public:

		logistic_regression (const double learning_rate = 0.1, const int32_t max_iter_ = 1000, const uint64_t seed = 0);

		virtual auto predict(const std::vector<double> & instance) const -> int32_t override;
		virtual auto predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<int32_t> override;

		virtual auto fit(const std::vector<std::vector<double> > & data, const std::vector<int32_t> & targets) -> void override;

};


} // end of models::regressors namespace

#endif
