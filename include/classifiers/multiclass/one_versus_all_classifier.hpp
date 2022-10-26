#ifndef ONE_VERSUS_ALL_CLASSIFIER_HPP
#define ONE_VERSUS_ALL_CLASSIFIER_HPP

#include <concepts>
#include "classifiers.hpp"

namespace cpp_ml::models::classifiers {

template <class T>
requires std::derived_from<T, classifier>
class one_versus_all_classifier : public classifier {
	private:
		T base_estimator_;
		std::vector<T> estimators_ {};

	public:

		one_versus_all_classifier(const T & estimator);
		virtual ~one_versus_all_classifier();

		virtual auto predict(const std::vector<double> & instance) const -> int32_t override;
		virtual auto predict_probabilities(const std::vector<double> & instance) const -> std::vector<double> override;
		virtual auto predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<int32_t> override;
		virtual auto predict_probabilities(const std::vector<std::vector<double>> & new_data) const -> std::vector<std::vector<double>> override;

		virtual auto fit(const std::vector<std::vector<double> > & data, const std::vector<int32_t> & targets) -> void override;


};


} // end cpp_ml::models::classifiers

#include "classifiers/multiclass/one_versus_all_classifier.tpp"

#endif
