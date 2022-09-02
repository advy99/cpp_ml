#ifndef TRANSFORMERS_HPP
#define TRANSFORMERS_HPP

#include <vector>
#include <string>
#include <cstdint>

namespace cpp_ml::models::transformers {

class target_transformer {
	private:

	public:

		virtual auto transform(const std::vector<std::string> & instance) const -> std::vector<int32_t> = 0;

		virtual auto fit(const std::vector<std::string> & targets) -> void = 0;


};

} // end models::transformers namespace



#endif
