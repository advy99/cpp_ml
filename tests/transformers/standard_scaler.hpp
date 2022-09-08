#ifndef STANDARD_SCALER_TESTS_HPP
#define STANDARD_SCALER_TESTS_HPP

#include <gtest/gtest.h>

#include "transformers/standard_scaler.hpp"

TEST (standard_scaler_test, check_means_and_var) {

	std::vector<std::vector<double>> data;

	data.push_back({2.4, -102.23, 2313.23, 54326.345, 0.34});
	data.push_back({34, 234.652, -3453.34, 231.215, -23.2});
	data.push_back({123.2, 2.62, -3.34, 2131.215, -233.2});

	cpp_ml::models::transformers::standard_scaler data_scaler;

	data_scaler.fit(data);
	auto data_scaled = data_scaler.transform(data);

	auto computed_means = data_scaler.get_means();
	auto computed_var = data_scaler.get_standard_deviations();


	std::vector<double> means {53.2, 45.014, -381.15, 18896.258333333333333333, -85.353333333333333};
	std::vector<double> std_dev {51.15101823685103, 140.7605742812951, 2369.3019630684475, 25064.859610028187, 104.98415922837542};

	for (std::size_t i = 0; i < means.size(); ++i) {
		EXPECT_DOUBLE_EQ(means[i], computed_means[i]) << "Failed at index " << i;
		EXPECT_DOUBLE_EQ(std_dev[i], computed_var[i]) << "Failed ay index " << i;
	}


}

TEST (standard_scaler_test, unique_value_to_zero) {

	std::vector<std::vector<double>> data;

	data.push_back({11.0, 2.0});

	cpp_ml::models::transformers::standard_scaler data_scaler;

	data_scaler.fit(data);
	auto data_scaled = data_scaler.transform(data);


	EXPECT_DOUBLE_EQ(data_scaled[0][0], 0.0);
	EXPECT_DOUBLE_EQ(data_scaled[0][1], 0.0);

}




#endif
