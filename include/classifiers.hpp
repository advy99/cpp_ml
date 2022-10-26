#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <vector>
#include <cstdint>
#include <set>

namespace cpp_ml::models::classifiers {

class classifier {
	protected:
		std::size_t num_classes_ {};

	public:
		virtual ~classifier() { }

		virtual auto predict(const std::vector<double> & instance) const -> int32_t = 0;

		virtual auto predict_probabilities(const std::vector<double> & instance) const -> std::vector<double> = 0;

		virtual auto predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<int32_t> = 0;
		virtual auto predict_probabilities(const std::vector<std::vector<double>> & new_data) const -> std::vector<std::vector<double>> = 0;

		virtual auto fit(const std::vector<std::vector<double> > & data, const std::vector<int32_t> & targets) -> void {
			(void) data;
			num_classes_ = std::set(targets.begin(), targets.end()).size();
		}


};

} // end models::classifiers namespace



#endif
