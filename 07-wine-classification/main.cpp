#include <iostream>
#include "ETL.h"


void test() {
	std::string dataset_file_path = ".\\wine\\wine.data";
	try
	{
	// 	ETL etl(argv[1], argv[2], argv[3]);
	//	std::vector<std::vector<std::string>> dataset = etl.readCSV();
	//	//ETL etl(dataset_file_path, ",", false);
	//	std::vector<std::vector<std::string>> data = etl.readCSV();

	//	for (auto row : data) {
	//		for (auto col : row) {
	//			std::cout << col << " ";
	//		}
	//		std::cout << std::endl;
	//	}
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}

int main(int argc, char* argv[]) {
	
	std::string dataset_file_path = ".\\wine\\wine.data";

	ETL etl(dataset_file_path, ",", false);

	std::vector<std::vector<std::string>> dataset = etl.readCSV();
	
	int rows = dataset.size();
	int cols = dataset[0].size();


	Eigen::MatrixXd dataMat = etl.CSVtoEigen(dataset, rows, cols);

	std::cout << dataMat << std::endl;

	return EXIT_SUCCESS;
}