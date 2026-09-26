#include "ETL.h"

#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>

std::vector<std::vector<std::string >> ETL::readCSV() {
	std::ifstream file(dataset);
	std::vector<std::vector<std::string>> dataString;
	std::string line = "";


	while (getline(file, line)) {
		std::vector<std::string> vec;
		boost::algorithm::split(vec, line, boost::is_any_of(delimiter));
		dataString.push_back(vec);
	}

	file.close();

	return dataString;
}


Eigen::MatrixXd ETL::CSVtoEigen(std::vector<std::vector<std::string>>  dataset, int rows, int  cols)
{

	if (header == true) {
		rows--;
	}

	Eigen::MatrixXd mat(cols, rows);


	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			mat(j, i) = atof(dataset[i][j].c_str());
		}
	}

	return mat.transpose();
}

/// <summary>
/// Z = (x - mue) / standard-deviation
/// </summary>
/// <param name="data"></param>
/// <returns></returns>

auto ETL::Mean(Eigen::MatrixXd data)-> decltype(data.colwise().mean()) {
	return data.colwise().mean();
}

auto ETL::Std(Eigen::MatrixXd data)-> decltype(((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt()) {
	return ((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt();
}

Eigen::MatrixXd ETL::Normalize(Eigen::MatrixXd data) {
	auto mean = Mean(data);
	Eigen::MatrixXd scaled_data = data.rowwise() - mean;
	auto std = Std(scaled_data);

	Eigen::MatrixXd norm = scaled_data.array().rowwise() / std;

	return norm;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ETL::TrainTestSplit(Eigen::MatrixXd data, float train_size) {

	int rows = data.rows();
	int train_rows = round(rows * train_size);
	int test_rows =rows - train_rows;

	Eigen::MatrixXd train = data.topRows(train_rows);

	Eigen::MatrixXd x_train = data.leftCols(data.cols() - 1);
	Eigen::MatrixXd y_train = data.rightCols(1);


	Eigen::MatrixXd test = data.bottomRows(test_rows);

	Eigen::MatrixXd x_test = test.leftCols(data.cols() - 1);
	Eigen::MatrixXd y_test = test.rightCols(1);

	return std::make_tuple(x_train, y_train, x_test, y_test);

}