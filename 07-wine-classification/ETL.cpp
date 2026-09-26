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

	if(header == true){
		rows--;
	}

	Eigen::MatrixXd mat(cols,rows);
	

	for (int i = 0; i < rows; i++ ) {
		for (int j = 0; j < cols; j++) {
			mat(j, i) = atof(dataset[i][j].c_str());
		}
	}

	return mat.transpose();
}
