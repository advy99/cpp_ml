#ifndef STANDARD_SCALER_HPP
#define STANDARD_SCALER_HPP

#include "transformers.hpp"

namespace cpp_ml::models::transformers {

class standard_scaler : public data_transformer {
	private:
		// a mean and standard_deviation per data column
		std::vector<double> means_;
		std::vector<double> standards_deviations_;

	public:

		standard_scaler();
		virtual ~standard_scaler() { }

		virtual auto transform(const std::vector<std::vector<double>> & data) const -> std::vector<std::vector<double>> override;

		virtual auto fit(const std::vector<std::vector<double>> & data) -> void override;

		auto get_means() const -> std::vector<double>;

		auto get_standard_deviations() const -> std::vector<double>;

};


} // end namespace models::transformers

#endif
