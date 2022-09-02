#include "transformers/label_encoder.hpp"

namespace cpp_ml::models::transformers {

label_encoder :: label_encoder () : num_unique_values_ {0}
{

}


auto label_encoder :: transform (const std::vector<std::string> & instances) const -> std::vector<int32_t> {
	std::vector<int32_t> result;

	for (const auto & value : instances) {
		result.emplace_back(transformation_.at(value));
	}

	return result;
}


auto label_encoder :: fit (const std::vector<std::string> & targets) -> void {

	num_unique_values_ = 0;
	transformation_ = std::map<std::string, int32_t>();

	for (const auto & element : targets) {
		if ( !transformation_.contains(element) ) {
			transformation_[element] = num_unique_values_;
			num_unique_values_ += 1;
		}
	}

}

} // end models::transformers namespace
