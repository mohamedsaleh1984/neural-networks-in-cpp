#include <iostream>
#include "ETL.h"
#include <string>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>

int main(int argc, char* argv[]) {
	
	std::string dataset_file_path = ".\\wine\\wine.data";

	ETL etl(dataset_file_path, ",", false);

	std::vector<std::vector<std::string>> dataset = etl.readCSV();
	
	int rows = dataset.size();
	int cols = dataset[0].size();


	Eigen::MatrixXd dataMat = etl.CSVtoEigen(dataset, rows, cols);

	Eigen::MatrixXd norm = etl.Normalize(dataMat);

	Eigen::MatrixXd x_train, y_train, x_test, y_test;
	std::tuple< Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> split_data = etl.TrainTestSplit(norm, 0.8);

	std::tie(x_train, y_train, x_test, y_test) = split_data;

	std::cout << "x-train " << x_train.rows() << std::endl;
	std::cout << "y-train " << y_train.cols() << std::endl;
	std::cout << "x-test " << x_test.rows() << std::endl;
	std::cout << "y-test " << y_test.cols() << std::endl;


	return EXIT_SUCCESS;
}