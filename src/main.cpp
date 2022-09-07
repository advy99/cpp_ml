#include <iostream>

// used library parts
#include "classifiers/knn_classifier.hpp"
#include "regressors/linear_regression.hpp"
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

int main(int argc, char ** argv) {

	if (argc != 3) {
		std::cout << "ERROR: This program must be launched with two parameters.\n"
					 << "\t" << argv[0] << " <data_file> <test_percentage>\n";
		std::exit(1);
	}

	std::string data_file = argv[1];
	double test_percentage = std::stod(std::string(argv[2]));


	// create a knn with k = 3 and using the euclidean distance
	classifiers::knn_classifier my_knn = classifiers::knn_classifier(3, distances::euclidean_distance);

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

	// fit the knn and predict the test dataset
	my_knn.fit(x_train, y_train);

	auto test_predictions = my_knn.predict(x_test);
	double test_accuracy = metrics::classification_metrics::accuracy_score(y_test, test_predictions);

	auto train_predictions = my_knn.predict(x_train);
	double train_accuracy = metrics::classification_metrics::accuracy_score(y_train, train_predictions);

	std::cout << "Train accuracy: " << train_accuracy << "\n"
				 << "Test accuracy: " << test_accuracy << "\n";


	std::cout << "\n" << "As a regression problem using linear regression:" << "\n";

	regressors::linear_regression my_linear_reg;


	std::vector<double> y_train_double;
	std::vector<double> y_test_double;

	for (const auto & value : y_train) {
		y_train_double.emplace_back(static_cast<double>(value));
	}

	for (const auto & value : y_test) {
		y_test_double.emplace_back(static_cast<double>(value));
	}

	my_linear_reg.fit(x_train, y_train_double);

	auto test_predictions_reg = my_linear_reg.predict(x_test);
	double test_accuracy_reg = metrics::regression_metrics::mean_squared_error(y_test_double, test_predictions_reg);

	auto train_predictions_reg = my_linear_reg.predict(x_train);
	double train_accuracy_reg = metrics::regression_metrics::mean_squared_error(y_train_double, train_predictions_reg);

	std::cout << "Train MSE: " << train_accuracy_reg << "\n"
				 << "Test MSE: " << test_accuracy_reg << "\n";



	return 0;
}
