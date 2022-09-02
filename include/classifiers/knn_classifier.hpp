#ifndef KNNCLASSIFIER_HPP
#define KNNCLASSIFIER_HPP

#include "classifiers.hpp"
#include <functional>

namespace cpp_ml::models::classifiers {

class knn_classifier : public classifier {
	private:
		std::vector<std::vector<double>> data_;
		std::vector<int32_t> targets_;
		std::function<double(const std::vector<double> &, const std::vector<double> &)> distance_function_;
		int32_t k_;

	public:

		knn_classifier(int32_t k, const std::function<double(const std::vector<double> &, const std::vector<double> &)> & distance_f);

		virtual auto predict(const std::vector<double> & instance) const -> int32_t override;
		virtual auto predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<int32_t> override;

		virtual auto fit(const std::vector<std::vector<double> > & data, const std::vector<int32_t> & targets) -> void override;

};


} // end of classifiers namespace

#endif
