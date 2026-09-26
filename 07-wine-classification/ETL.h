#pragma once
#ifndef ETL_H
#define ETL_H

#include <iostream>
#include <fstream>
#include <Eigen/Dense>

class ETL {
	std::string dataset;
	std::string delimiter;
	bool header;

public:
	ETL(std::string data, std::string separaator, bool head) :
		dataset(data), delimiter(separaator), header(head) {

	}
	std::vector<std::vector<std::string >> readCSV();
	Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>>  dataset, int rows, int  cols);
	Eigen::MatrixXd Normalize(Eigen::MatrixXd data);
	auto Mean(Eigen::MatrixXd data)-> decltype(data.colwise().mean());
	auto Std(Eigen::MatrixXd data)-> decltype(((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt());
	std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd data, float train_size);
};

#endif 