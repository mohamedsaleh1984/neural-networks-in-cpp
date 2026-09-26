#include "ETL.h"

#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>


/// <summary>
/// Reads the configured CSV dataset from disk and splits each line into
/// individual string fields using the configured delimiter.
/// </summary>
/// <returns>A vector of rows, where each row is a vector of string fields.</returns>
std::vector<std::vector<std::string >> ETL::readCSV() {
	std::ifstream file(dataset);
	std::vector<std::vector<std::string>> dataString;
	std::string line = "";

	// Read the file line by line.
	while (getline(file, line)) {
		std::vector<std::string> vec;
		// Split the current line on the delimiter (e.g. ",") into fields.
		boost::algorithm::split(vec, line, boost::is_any_of(delimiter));
		dataString.push_back(vec);
	}

	file.close();

	return dataString;
}

/// <summary>
/// Converts the raw string data read from the CSV file into an Eigen matrix
/// of doubles. If the CSV contains a header row it is skipped.
/// </summary>
/// <param name="dataset">Raw string data as returned by readCSV().</param>
/// <param name="rows">Total number of rows in the dataset (including the header row, if any).</param>
/// <param name="cols">Number of columns (features + label) in the dataset.</param>
/// <returns>A rows x cols matrix of the numeric values from the dataset.</returns>
Eigen::MatrixXd ETL::CSVtoEigen(std::vector<std::vector<std::string>>  dataset, int rows, int  cols)
{

	// Drop the header row from the row count so it isn't converted to a number.
	if (header == true) {
		rows--;
	}

	// Build the matrix transposed (cols x rows) first so that each
	// source row can be written contiguously as a matrix column.
	Eigen::MatrixXd mat(cols, rows);


	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			// Convert the string field to a double and store it.
			mat(j, i) = atof(dataset[i][j].c_str());
		}
	}

	// Transpose back to the natural rows x cols layout before returning.
	return mat.transpose();
}

/// <summary>
/// Computes the per-column (per-feature) mean of the data.
/// Used for standardization: Z = (x - mue) / standard-deviation
/// </summary>
/// <param name="data">The data matrix whose columns are features.</param>
/// <returns>A row vector containing the mean of each column.</returns>
auto ETL::Mean(Eigen::MatrixXd data)-> decltype(data.colwise().mean()) {
	return data.colwise().mean();
}


/// <summary>
/// Computes the per-column sample standard deviation of the data
/// (sum of squared values divided by N - 1, then square-rooted).
/// </summary>
/// <param name="data">The data matrix whose columns are features.</param>
/// <returns>A row vector containing the standard deviation of each column.</returns>
auto ETL::Std(Eigen::MatrixXd data)-> decltype(((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt()) {
	return ((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt();
}

/// <summary>
/// Standardizes (z-score normalizes) the data so each feature column
/// has zero mean and unit variance: norm = (x - mean) / std.
/// </summary>
/// <param name="data">The raw data matrix to normalize.</param>
/// <returns>The normalized data matrix.</returns>
Eigen::MatrixXd ETL::Normalize(Eigen::MatrixXd data) {
	// Subtract the per-feature mean from every row (center the data).
	auto mean = Mean(data);
	Eigen::MatrixXd scaled_data = data.rowwise() - mean;

	// Compute the standard deviation of the centered data.
	auto std = Std(scaled_data);

	// Divide each feature by its standard deviation.
	Eigen::MatrixXd norm = scaled_data.array().rowwise() / std;

	return norm;
}

/// <summary>
/// Splits the dataset into training and test sets along the row axis,
/// then separates each into feature matrices (X) and label vectors (y).
/// The label is assumed to be the last column of the data.
/// </summary>
/// <param name="data">The full dataset (features + label in the last column).</param>
/// <param name="train_size">Fraction of rows to use for training (e.g. 0.8).</param>
/// <returns>A tuple of (x_train, y_train, x_test, y_test).</returns>

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ETL::TrainTestSplit(Eigen::MatrixXd data, float train_size) {
	int rows = data.rows();
	int train_rows = round(rows * train_size);
	int test_rows =rows - train_rows;

	// Training rows are taken from the top of the dataset.
	Eigen::MatrixXd train = data.topRows(train_rows);

	// Split training data into features (all but last column) and label (last column).
	Eigen::MatrixXd x_train = data.leftCols(data.cols() - 1);
	Eigen::MatrixXd y_train = data.rightCols(1);

	// Test rows are taken from the bottom of the dataset.
	Eigen::MatrixXd test = data.bottomRows(test_rows);

	// Split test data into features and label.
	Eigen::MatrixXd x_test = test.leftCols(data.cols() - 1);
	Eigen::MatrixXd y_test = test.rightCols(1);

	return std::make_tuple(x_train, y_train, x_test, y_test);

}