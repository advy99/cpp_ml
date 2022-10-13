#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <vector>
#include <cstdint>

namespace cpp_ml::models::classifiers {

class classifier {
	private:

	public:
		virtual ~classifier() { }

		virtual auto predict(const std::vector<double> & instance) const -> int32_t = 0;

		virtual auto predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<int32_t> = 0;

		virtual auto fit(const std::vector<std::vector<double> > & data, const std::vector<int32_t> & targets) -> void = 0;


};

} // end models::classifiers namespace



#endif
