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

};

#endif 