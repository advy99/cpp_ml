#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <opencv2/core.hpp>

template <typename T> concept arithmetic = std::is_arithmetic_v<T>;

namespace cpp_ml::math::matrix {

template <typename T>
auto transpose (const std::vector<std::vector<T>> & matrix) -> std::vector<std::vector<T>> {

	std::vector<std::vector<T>> transposed ( matrix[0].size(), std::vector<T>(matrix.size()) );

	for (std::size_t i = 0; i < matrix[0].size(); ++i) {
		for (std::size_t j = 0; j < matrix.size(); ++j) {
			transposed[i][j] = matrix[j][i];
		}
	}

	return transposed;

}


template <typename T>
requires arithmetic<T>
auto dot (const std::vector<std::vector<T>> & matrix_a,
			 const std::vector<std::vector<T>> & matrix_b) -> std::vector<std::vector<T>> {

	std::vector<std::vector<T>> result ( matrix_a.size(), std::vector<T>(matrix_b[0].size()) );

	for (std::size_t i = 0; i < matrix_a.size(); ++i) {
		for (std::size_t j = 0; j < matrix_b[0].size(); ++j) {
			result[i][j] = 0.0;

			for (std::size_t k = 0; k < matrix_a[0].size(); ++k) {
				result[i][j] += matrix_a[i][k] * matrix_b[k][j];
			}
		}
	}

	return result;

}


template <typename T>
requires arithmetic<T>
auto invert (const std::vector<std::vector<T>> & matrix) -> std::vector<std::vector<T>> {
	cv::Mat matrix_cv(matrix.size(), matrix.at(0).size(), CV_64FC1);
	for(int i = 0; i < matrix_cv.rows; ++i) {
		for(int j = 0; j < matrix_cv.cols ; ++j) {
			matrix_cv.at<double>(i, j) = matrix.at(i).at(j);
		}
	}

	cv::Mat result_cv = matrix_cv.inv(cv::DECOMP_SVD);

	std::vector<std::vector<T>> result (matrix.at(0).size(), std::vector<T>(matrix.size()));

	for(int i = 0; i < result_cv.rows; ++i) {
		for(int j = 0; j < result_cv.cols ; ++j) {
			result[i][j] = result_cv.at<double>(i, j);
		}
	}

	return result;


}



} // ends math::matrix namespace



#endif
