
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
using namespace std;

namespace data_layer {
	/// <summary>
	/// iris entity
	/// </summary>
	struct iris {
		std::vector<double> features; // [sepal_length, sepal_width, petal_length, petal_width]
		std::string label;            // Species name
	};
}


namespace classification {
	using namespace data_layer;
	/// <summary>
	/// iris class data range
	/// </summary>
	vector<vector<int>>  data_range = { {0,49},{50,99},{100,149} };

	/// <summary>
	/// Constructs a training dataset by selecting specific entries from an iris dataset and converting each to a numeric vector.
	/// </summary>
	/// <param name="dataset">A vector of iris objects representing the full dataset. Elements at indices 0–39, 50–89, and 100–139 are selected and converted for training.</param>
	/// <returns>A vector of numeric feature vectors (vector<vector<double>>), where each inner vector is the result of irisToDblVector for a selected iris instance.</returns>
	vector<iris> get_training_dataset(const vector<iris>& dataset) {
		vector<iris> training;
		for (int j = 0; j < data_range.size(); j++) {
			for (int i = 0; i < dataset.size(); i++) {
				if (i >= data_range[j][0] && i <= (data_range[j][1] - 10)) {
					training.push_back(dataset[i]);
				}
			}
		}

		return training;
	}



	/// <summary>
	/// Builds a test dataset by selecting specific entries from an iris dataset and converting each selected record to a vector<double>.
	/// </summary>
	/// <param name="dataset">The input collection of iris records. The function selects records at indices 40–48 (i > 39 && i < 49), 90–99 (i > 89 && i <= 99), and all indices greater than 139 (i > 139), converting each selected iris to a vector<double> using irisToDblVector.</param>
	/// <returns>A vector of vectors of doubles where each inner vector is the result of converting a selected iris record (from the specified index ranges) to numeric features.</returns>
	vector<iris> get_test_dataset(const vector<iris>& dataset) {
		vector<iris>  test;
		for (int j = 0; j < data_range.size(); j++) {
			for (int i = 0; i < dataset.size(); i++) {

				if (i >= (data_range[j][0] + 40) && i <= data_range[j][1]) {
					test.push_back(dataset[i]);
				}
			}
		}
		return test;
	}


	vector<vector<iris>> getTraningTestData(const vector<iris>& dataset) {
		vector<vector<iris>> ret;

		vector<iris> t = get_training_dataset(dataset);
		vector<iris> tes = get_test_dataset(dataset);
		
		random_shuffle(t.begin(), t.end());
		random_shuffle(tes.begin(), tes.end());
		
		ret.push_back(t);
		ret.push_back(tes);

		return ret;
	}

	// Calculate Euclidean Distance between two feature vectors
	double calculateDistance(const std::vector<double>& a, const std::vector<double>& b) {
		double sum = 0.0;
		for (size_t i = 0; i < a.size(); ++i) {
			sum += std::pow(a[i] - b[i], 2);
		}
		return std::sqrt(sum);
	}

	// KNN Classifier Implementation
	std::string predictKNN(const std::vector<data_layer::iris>& trainData, const std::vector<double>& testFeatures) {
		int k = 3; // Number of neighbors
		// Pairs of (distance, label)
		std::vector<std::pair<double, std::string>> distances;

		 
		for (const auto& trainSample : trainData) {
			double dist = calculateDistance(trainSample.features, testFeatures);
			distances.push_back({ dist, trainSample.label });
		}

		// Sort neighbors based on closest distance
		std::sort(distances.begin(), distances.end());

		// Count votes among the top K nearest neighbors
		std::map<std::string, int> classVotes;
		for (int i = 0; i < k; ++i) {
			classVotes[distances[i].second]++;
		}

		// Return the class label with the highest vote
		std::string bestClass;
		int maxVotes = -1;
		for (const auto& vote : classVotes) {
			if (vote.second > maxVotes) {
				maxVotes = vote.second;
				bestClass = vote.first;
			}
		}
		return bestClass;
	}
};

int main()
{
	try
	{
		std::ifstream file("iris.csv");
		if (!file.is_open()) {
			std::cerr << "Error: Could not open iris.csv file." << std::endl;
			return 1;
		}

		std::vector<data_layer::iris> dataset;
		std::string line;

		// Load data from CSV
		while (std::getline(file, line)) {
			if (line.empty()) continue;

			std::stringstream ss(line);
			std::string token;
			data_layer::iris sample;

			for (int i = 0; i < 4; ++i) {
				std::getline(ss, token, ',');
				sample.features.push_back(std::stod(token));
			}
			std::getline(ss, token, ',');
			sample.label = token;

			dataset.push_back(sample);
		}
		file.close();

		std::vector<std::vector<data_layer::iris>> daSet = classification::getTraningTestData(dataset);
		
		std::vector<data_layer::iris> trainData = daSet[0];
		std::vector<data_layer::iris> testData = daSet[1];

		std::cout << "Dataset Loaded. Train instances: " << trainData.size() << " | Test instances: " << testData.size() << "\n\n";

		// Evaluate the model on the test data
		int correctPredictions = 0;

		std::cout << "--- Test Set Evaluation ---\n";
		for (const auto& testSample : testData) {
			std::string prediction = classification::predictKNN(trainData, testSample.features);

			std::cout << "Actual: " << testSample.label << " | Predicted: " << prediction;

			if (prediction == testSample.label) {
				std::cout << " [Correct]\n";
				correctPredictions++;
			}
			else {
				std::cout << " [Incorrect]\n";
			}
		}

		// Calculate accuracy
		double accuracy = (double)correctPredictions / testData.size() * 100.0;
		std::cout << "\n====================================\n";
		std::cout << "Model Accuracy: " << accuracy << "%\n";
		std::cout << "====================================\n";

		

	}
	catch (const std::exception&)
	{

	}


	return 0;
}
