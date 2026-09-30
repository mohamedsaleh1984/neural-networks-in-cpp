
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

	// Calculate Euclidean Distance between two feature vectors
	double calculateDistance(const std::vector<double>& a, const std::vector<double>& b) {
		double sum = 0.0;
		for (size_t i = 0; i < a.size(); ++i) {
			sum += std::pow(a[i] - b[i], 2);
		}
		return std::sqrt(sum);
	}

	// KNN Classifier Implementation
	std::string predictKNN(const std::vector<data_layer::iris>& trainData, const std::vector<double>& testFeatures, int k) {
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

		// Naive Train/Test Split (e.g., Use every 5th element for testing, rest for training)
		std::vector<data_layer::iris> trainData;
		std::vector<data_layer::iris> testData;

		for (size_t i = 0; i < dataset.size(); ++i) {
			if (i % 5 == 0) {
				testData.push_back(dataset[i]); // 20% Testing data
			}
			else {
				trainData.push_back(dataset[i]); // 80% Training data
			}
		}

		std::cout << "Dataset Loaded. Train instances: " << trainData.size()
			<< " | Test instances: " << testData.size() << "\n\n";

		// Evaluate the model on the test data
		int k = 3; // Number of neighbors
		int correctPredictions = 0;

		std::cout << "--- Test Set Evaluation ---\n";
		for (const auto& testSample : testData) {
			std::string prediction = classification::predictKNN(trainData, testSample.features, k);

			std::cout << "Actual: " << testSample.label
				<< " | Predicted: " << prediction;

			if (prediction == testSample.label) {
				std::cout << " [✓]\n";
				correctPredictions++;
			}
			else {
				std::cout << " [✗]\n";
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
