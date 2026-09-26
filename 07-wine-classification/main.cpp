#include <iostream>
#include "ETL.h"



int main() {

	std::string dataset_file_path = ".\\wine\\wine.data";
	try
	{
		ETL etl(dataset_file_path, ",", false);
		std::vector<std::vector<std::string>> data = etl.readCSV();

		for (auto row : data) {
			for (auto col : row) {
				std::cout << col << " ";
			}
			std::cout << std::endl;
		}
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	
	return 0;
}