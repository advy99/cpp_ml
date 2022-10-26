#include <iostream>

// used library parts
#include "classifiers/knn_classifier.hpp"
#include "classifiers/logistic_regression.hpp"
#include "regressors/linear_regression.hpp"
#include "regressors/knn_regressor.hpp"
#include "regressors/polynomial_regression.hpp"
#include "transformers/label_encoder.hpp"
#include "transformers/standard_scaler.hpp"
#include "math/distances.hpp"
#include "datasets/reading.hpp"
#include "datasets/split.hpp"
#include "metrics/classification_metrics.hpp"
#include "metrics/regression_metrics.hpp"

// local namespaces for simplicity
namespace classifiers = cpp_ml::models::classifiers;
namespace regressors = cpp_ml::models::regressors;
namespace transformers = cpp_ml::models::transformers;
namespace distances = cpp_ml::math::distances;
namespace datasets = cpp_ml::datasets;
namespace metrics = cpp_ml::metrics;



void test_knn_classifier(const std::vector<std::vector<double>> & x_train, const std::vector<int32_t> & y_train,
				  const std::vector<std::vector<double>> & x_test, const std::vector<int32_t> & y_test) {

	for (std::size_t val = 1; val < 11; val += 2){
		std::cout << "\nNow with KNN classifier with k = " << val << " \n";
		// create a knn with k = 3 and using the euclidean distance
		classifiers::knn_classifier my_knn = classifiers::knn_classifier(val, distances::euclidean_distance);

		// fit the knn and predict the test dataset
		my_knn.fit(x_train, y_train);

		auto test_predictions = my_knn.predict(x_test);
		double test_accuracy = metrics::classification_metrics::accuracy_score(y_test, test_predictions);

		auto train_predictions = my_knn.predict(x_train);
		double train_accuracy = metrics::classification_metrics::accuracy_score(y_train, train_predictions);

		std::cout << "\tTrain accuracy: " << train_accuracy << "\n"
					 << "\tTest accuracy: " << test_accuracy << "\n";

	}

}


void test_logistic_regression(const std::vector<std::vector<double>> & x_train, const std::vector<int32_t> & y_train,
				  const std::vector<std::vector<double>> & x_test, const std::vector<int32_t> & y_test) {

	std::cout << "\nNow with logistic regression:\n";
	// create a knn with k = 3 and using the euclidean distance
	classifiers::logistic_regression my_log_reg;

	// fit the knn and predict the test dataset
	my_log_reg.fit(x_train, y_train);

	auto test_predictions = my_log_reg.predict(x_test);
	double test_accuracy = metrics::classification_metrics::accuracy_score(y_test, test_predictions);

	auto train_predictions = my_log_reg.predict(x_train);
	double train_accuracy = metrics::classification_metrics::accuracy_score(y_train, train_predictions);

	std::cout << "\tTrain accuracy: " << train_accuracy << "\n"
				 << "\tTest accuracy: " << test_accuracy << "\n";



}


void test_knn_regressor(const std::vector<std::vector<double>> & x_train, const std::vector<double> & y_train,
				  				const std::vector<std::vector<double>> & x_test, const std::vector<double> & y_test) {

	for (std::size_t val = 1; val < 11; val += 2){
		std::cout << "\nNow with KNN regressor with k = " << val << " \n";

		regressors::knn_regressor my_knn_regressor(val, distances::euclidean_distance);

		my_knn_regressor.fit(x_train, y_train);

		auto test_predictions_poly_reg = my_knn_regressor.predict(x_test);
		double test_accuracy_poly_reg = metrics::regression_metrics::mean_squared_error(y_test, test_predictions_poly_reg);

		auto train_predictions_poly_reg = my_knn_regressor.predict(x_train);
		double train_accuracy_poly_reg = metrics::regression_metrics::mean_squared_error(y_train, train_predictions_poly_reg);

		std::cout << "\tTrain MSE: " << train_accuracy_poly_reg << "\n"
					 << "\tTest MSE: " << test_accuracy_poly_reg << "\n";
	}
}

void test_polynomial_regressor(const std::vector<std::vector<double>> & x_train, const std::vector<double> & y_train,
				  						 const std::vector<std::vector<double>> & x_test, const std::vector<double> & y_test) {

	for (std::size_t degree = 1; degree <= 3; ++degree) {

		std::cout << "\nNow with Polynomial regression of degree = " << degree << " \n";

		regressors::polynomial_regression my_poly_regressor(degree);

		my_poly_regressor.fit(x_train, y_train);

		auto test_predictions_poly_reg = my_poly_regressor.predict(x_test);
		double test_accuracy_poly_reg = metrics::regression_metrics::mean_squared_error(y_test, test_predictions_poly_reg);

		auto train_predictions_poly_reg = my_poly_regressor.predict(x_train);
		double train_accuracy_poly_reg = metrics::regression_metrics::mean_squared_error(y_train, train_predictions_poly_reg);

		std::cout << "\tTrain MSE: " << train_accuracy_poly_reg << "\n"
					 << "\tTest MSE: " << test_accuracy_poly_reg << "\n";

	}
}

int main(int argc, char ** argv) {

	if (argc != 3) {
		std::cout << "ERROR: This program must be launched with two parameters.\n"
					 << "\t" << argv[0] << " <data_file> <test_percentage>\n";
		std::exit(1);
	}

	std::string data_file = argv[1];
	double test_percentage = std::stod(std::string(argv[2]));



	transformers::label_encoder my_encoder = transformers::label_encoder();
	transformers::standard_scaler my_std_scaler = transformers::standard_scaler();

	// read the dataset
	std::pair dataset = datasets::reading::read_x_y_from_csv<double, std::string>(data_file, true, '#', ',');

	// fit the encoder and transform the data
	my_encoder.fit(dataset.second);

	auto original_features = dataset.first;
	auto targets = my_encoder.transform(dataset.second);

	my_std_scaler.fit(original_features);
	auto features = my_std_scaler.transform(original_features);


	// split in train/test
	std::tuple dataset_splits = datasets::split::train_test_split(features, targets, test_percentage);

	auto x_train = std::get<0>(dataset_splits);
	auto y_train = std::get<1>(dataset_splits);
	auto x_test = std::get<2>(dataset_splits);
	auto y_test = std::get<3>(dataset_splits);


	std::cout << "Complete dataset length: " << dataset.second.size() << "\n"
				 << "Train dataset length: " << y_train.size() << "\n"
				 << "Test dataset length: " << y_test.size() << "\n"
				 << "Percentage of test data: " << test_percentage << "\n";

	test_knn_classifier(x_train, y_train, x_test, y_test);


	// convert to binary problem to test logistic regression
	std::vector<int32_t> y_train_binary;
	std::vector<int32_t> y_test_binary;

	for (const auto & value : y_train) {
		int32_t val_class = 0;
		if (value > 1) {
			val_class = 1;
		}

		y_train_binary.push_back(val_class);
	}

	for (const auto & value : y_test) {
		int32_t val_class = 0;
		if (value > 1) {
			val_class = 1;
		}

		y_test_binary.push_back(val_class);
	}

	test_logistic_regression(x_train, y_train_binary, x_test, y_test_binary);

	std::vector<double> y_train_double;
	std::vector<double> y_test_double;

	for (const auto & value : y_train) {
		y_train_double.emplace_back(static_cast<double>(value));
	}

	for (const auto & value : y_test) {
		y_test_double.emplace_back(static_cast<double>(value));
	}

	test_knn_regressor(x_train, y_train_double, x_test, y_test_double);
	test_polynomial_regressor(x_train, y_train_double, x_test, y_test_double);

	return 0;
}
