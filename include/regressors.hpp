#ifndef REGRESSORS_HPP
#define REGRESSORS_HPP

#include <vector>
#include <cstdint>

namespace cpp_ml::models::regressors {

class regressor {
	private:

	public:
		virtual ~regressor () { }

		virtual auto predict(const std::vector<double> & instance) const -> double = 0;

		virtual auto predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<double> = 0;

		virtual auto fit(const std::vector<std::vector<double> > & data, const std::vector<double> & targets) -> void = 0;


};

} // end models::regressors namespace



#endif
