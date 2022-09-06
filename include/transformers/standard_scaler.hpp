#ifndef STANDARD_SCALER_HPP
#define STANDARD_SCALER_HPP

#include "transformers.hpp"

namespace cpp_ml::models::transformers {

class standard_scaler : public data_transformer {
	private:
		// a mean and standard_deviation per data column
		std::vector<double> mean_;
		std::vector<double> standard_deviation_;

	public:

		standard_scaler();

		virtual auto transform(const std::vector<std::vector<double>> & data) const -> std::vector<std::vector<double>> override;

		virtual auto fit(const std::vector<std::vector<double>> & data) -> void override;


};


} // end namespace models::transformers

#endif
