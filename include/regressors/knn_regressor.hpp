#ifndef KNNREGRESSOR_HPP
#define KNNREGRESSOR_HPP

#include "regressors.hpp"
#include <functional>

namespace cpp_ml::models::regressors {

class knn_regressor : public regressor {
	private:
		std::vector<std::vector<double>> data_ {};
		std::vector<double> targets_ {};
		std::function<double(const std::vector<double> &, const std::vector<double> &)> distance_function_;
		size_t k_;

	public:

		knn_regressor(size_t k, const std::function<double(const std::vector<double> &, const std::vector<double> &)> & distance_f);
		virtual ~knn_regressor();

		virtual auto predict(const std::vector<double> & instance) const -> double override;
		virtual auto predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<double> override;

		virtual auto fit(const std::vector<std::vector<double> > & data, const std::vector<double> & targets) -> void override;

};


} // end of models::regressors namespace

#endif
