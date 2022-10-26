#ifndef LABEL_ENCODER_HPP
#define LABEL_ENCODER_HPP


#include <map>
#include <vector>
#include "transformers.hpp"

namespace cpp_ml::models::transformers {

class label_encoder : public target_transformer {

	private:
		int32_t num_unique_values_;
		std::map<std::string, int32_t> transformation_ {};

	public:

		label_encoder();
		virtual ~label_encoder();

		virtual auto transform(const std::vector<std::string> & instance) const -> std::vector<int32_t> override;

		virtual auto fit(const std::vector<std::string> & targets) -> void override;

};

} // end models::transformers namespace

#endif
