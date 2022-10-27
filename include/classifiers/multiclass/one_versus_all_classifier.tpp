namespace cpp_ml::models::classifiers {

template <class T>
requires std::derived_from<T, classifier>
one_versus_all_classifier<T> :: one_versus_all_classifier (const T & estimator)
	:base_estimator_ (estimator)
{}

template <class T>
requires std::derived_from<T, classifier>
one_versus_all_classifier<T> :: ~one_versus_all_classifier() {}

template <class T>
requires std::derived_from<T, classifier>
auto one_versus_all_classifier<T> :: predict(const std::vector<double> & instance) const -> int32_t {

	auto probabilities = predict_probabilities(instance);

	auto iterator_to_max_item = std::max_element(probabilities.begin(), probabilities.end());

	int32_t prediction = static_cast<int32_t>(std::distance(probabilities.begin(), iterator_to_max_item));

	return prediction;

}


template <class T>
requires std::derived_from<T, classifier>
auto one_versus_all_classifier<T> :: predict_probabilities(const std::vector<double> & instance) const -> std::vector<double> {

	std::vector<double> probabilities (num_classes_, 0.0);

	for (std::size_t i = 0; i < estimators_.size(); ++i) {
		probabilities[i] = estimators_[i].predict_probabilities(instance)[1];
	}

	// TODO: Apply min_max_scaler to probabilities

	return probabilities;
}



template <class T>
requires std::derived_from<T, classifier>
auto one_versus_all_classifier<T> :: predict(const std::vector<std::vector<double> > & new_data) const -> std::vector<int32_t> {
	std::vector<int32_t> predictions;

	for (const auto & row : new_data) {
		predictions.push_back( predict(row) );
	}

	return predictions;

}

template <class T>
requires std::derived_from<T, classifier>
auto one_versus_all_classifier<T> :: fit(const std::vector<std::vector<double> > & data, const std::vector<int32_t> & targets) -> void {

	this->classifier::fit(data, targets);

	estimators_.resize(num_classes_);

	for (std::size_t i = 0; i < num_classes_; ++i) {
		std::vector<int32_t> new_targets;
		int32_t value = 0;

		for (const auto & target : targets) {
			if (target == static_cast<int32_t>(i)) {
				value = 1;
			} else {
				value = 0;
			}
			new_targets.push_back(value);
		}

		estimators_[i] = base_estimator_;
		estimators_[i].fit(data, new_targets);

	}

}

template <class T>
requires std::derived_from<T, classifier>
auto one_versus_all_classifier<T> :: predict_probabilities(const std::vector<std::vector<double>> & new_data) const -> std::vector<std::vector<double>> {
	std::vector<std::vector<double>> predictions;

	predictions.reserve(new_data.size());

	for (const auto & data : new_data ) {
		predictions.push_back(predict_probabilities(data));
	}

	return predictions;


}



} // end cpp_ml::models::classifiers
