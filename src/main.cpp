#include <iostream>

// used library parts
#include "classifiers/knn_classifier.hpp"
#include "transformers/label_encoder.hpp"
#include "math/distances.hpp"
#include "datasets/reading.hpp"
#include "datasets/split.hpp"
#include "metrics/classification_metrics.hpp"

// local namespaces for simplicity
namespace classifiers = cpp_ml::models::classifiers;
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

	// read the dataset
	std::pair dataset = datasets::reading::read_x_y_from_csv<double, std::string>(data_file, true, '#', ',');

	// fit the encoder and transform the data
	my_encoder.fit(dataset.second);

	auto features = dataset.first;
	auto targets = my_encoder.transform(dataset.second);

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

	return 0;
}
