#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <vector>
#include <map>
#include <cerrno>  // For errno
#include <cstring>  // For std::strerror
using namespace std;

#define _CRT_SECURE_NO_WARNINGS

/*
   -- The attributes are (dontated by Riccardo Leardi,
	riclea@anchem.unige.it )
	1) Alcohol
	2) Malic acid
	3) Ash
	4) Alcalinity of ash
	5) Magnesium
	6) Total phenols
	7) Flavanoids
	8) Nonflavanoid phenols
	9) Proanthocyanins
	10)Color intensity
	11)Hue
	12)OD280/OD315 of diluted wines
	13)Proline
*/

namespace classification {
	class Wine {
	public:
		Wine() = default;
		// Constructor that takes a line from the CSV file and parses it into a Wine object
		Wine(const std::string& line) {
			std::vector<std::string> tokens;
			boost::split(tokens, line, boost::is_any_of(","));
			if (tokens.size() != 14) {
				throw std::runtime_error("Invalid number of attributes in line: " + line);
			}
			label = std::stoi(tokens[0]);
			for (size_t i = 1; i < tokens.size(); ++i) {
				attributes.push_back(std::stod(tokens[i]));
			}
		}

		// Getters for label and attributes
		int getLabel() const {
			return label;
		}

		// Get the attributes of the wine sample
		const std::vector<double>& getAttributes() const {
			return attributes;
		}
	private:
		int label;
		std::vector<double> attributes;
	};

	class dataReader {
	private:
		std::string filename;
		std::string delimiter;
		bool header;
	public:
		/// <summary>
		/// Constructor for the dataReader class.
		/// </summary>
		/// <param name="file"></param>
		/// <param name="del"></param>
		/// <param name="head"></param>
		dataReader(const std::string& file, const std::string& del, bool head) :
			filename(file), delimiter(del), header(head) {
		}

		/// <summary>
		/// Reads the CSV file and returns a vector of Wine objects.
		/// </summary>
		/// <returns></returns>
		vector<Wine> readCSV() {
			ifstream file(filename);
			vector<Wine> wines = {};

			try
			{
				if (!file.is_open()) {
					char* szBuffer = new char[256];
					strerror_s(szBuffer, 256, errno);
					// Print the system-level error reason
					cerr << "Error opening file: " 
						 << szBuffer
						 << endl;
					return wines;
				}
			}
			catch (const std::exception& ex)
			{
				cout << "Exception caught while opening the file: " << endl;
				cout << ex.what() << endl;
			}

			try
			{
			
				std::string line;
				if (header) {
					std::getline(file, line); // Skip header
				}

				// Read each line and create Wine objects
				while (std::getline(file, line)) {
					try {
						Wine xwine(line);
						wines.push_back(xwine);
					}
					catch (const std::exception& e) {
						std::cerr << "Error parsing line: " << line << ". " << e.what() << std::endl;
					}
				}

			}
			catch (const std::exception&)
			{

			}
			return wines;
		}

		template<typename T>
		void append(vector<T>& to, vector<T> from)
		{
			if (to.size() == 0) {
				to.insert(to.begin(), from.begin(), from.end());
			}
			else {
			//	to.resize(to.size() + from.size() + 1);
				to.insert(to.begin() + to.size(), from.begin(), from.end());
			}
			to.shrink_to_fit();
			cout << to.size() << endl;
		}
		/// <summary>
		/// Splits the dataset into training and testing sets based on the specified ratio.
		/// </summary>
		/// <param name="data"></param>
		/// <param name="trainRatio"></param>
		/// <returns></returns>
		vector<vector<Wine>> splitData(const std::vector<Wine>& dataset) {
			vector<vector<int>>  data_range = { {0,58},{59,128},{129,177} };

			vector<Wine> trainData;
			vector<Wine> testData;

			// Train-test split logic based on the data_range
			for (int i = 0; i < data_range.size(); i++)
			{
				vector<Wine> tmp(dataset.begin()+ data_range[i][0],dataset.begin() + data_range[i][1]-10);
				append(trainData, tmp);

				vector<Wine> xtmp(dataset.begin() + data_range[i][1] - 10,dataset.begin() + data_range[i][1]);
				append(testData, xtmp);
			}
			
			return { trainData, testData };
		}
	};



	class Classifier {
	public:
		/// <summary>
		// Calculate Euclidean Distance between two feature vectors
		/// <param name="a">First feature vector</param>
		/// <param name="b">Second feature vector</param>
		///	<returns>Euclidean distance as a double</returns>
		/// </summary>
		double calculateDistance(const std::vector<double>& a,
			const std::vector<double>& b) {
			double sum = 0.0;
			for (size_t i = 0; i < a.size(); ++i) {
				sum += std::pow(a[i] - b[i], 2);
			}
			return std::sqrt(sum);
		}

		// KNN Classifier Implementation
		int predictKNN(const std::vector<Wine>& trainData, const Wine& testFeatures) {
			int k = 3; // Number of neighbors

			// Pairs of (distance, label)
			std::vector<std::pair<int, int>> distances;

			// Calculate distances from the test sample to all training samples 
			for (const auto& trainSample : trainData) {
				double dist = calculateDistance(trainSample.getAttributes(), testFeatures.getAttributes());
				distances.push_back({ dist, trainSample.getLabel() });
			}

			// Sort neighbors based on closest distance
			std::sort(distances.begin(), distances.end());

			// Count votes among the top K nearest neighbors
			std::map<int, int> classVotes;
			for (int i = 0; i < k; ++i) {
				classVotes[distances[i].second]++;
			}

			// Return the class label with the highest vote
			int bestClass;
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
}


int main(int argc, char* argv[]) {

	const double trainRatio = 0.8; // 80% training, 20% testing
	using namespace classification;

	cout << "Wine Classification using KNN" << endl;
	cout << "Using training ratio: " << trainRatio * 100 << "%" << endl;
	cout << "Reading dataset from: wine.data" << endl;
	// Read the wine dataset and split it into training and testing sets
	dataReader reader("wine.data", ",", false);
	vector<Wine> wines = reader.readCSV();

	cout << "Total samples read: " << wines.size() << endl;

	// Split the dataset into training and testing sets
	cout << "Splitting dataset into training and testing sets..." << endl;
	vector<vector<Wine>> splitData = reader.splitData(wines);
	vector<Wine> trainData = splitData[0];
	vector<Wine> testData = splitData[1];

	// Evaluate the classifier on the test set
	cout << "Evaluating classifier on test set..." << endl;
	Classifier classifier;
	int correctPredictions = 0;
	cout << "Total test samples: " << testData.size() << endl;

	for (const auto& testSample : testData) {
		cout << "Testing sample with label: " << testSample.getLabel() << endl;
		int predictedLabel = classifier.predictKNN(trainData, testSample);
		if (predictedLabel == testSample.getLabel()) {
			correctPredictions++;
		}
	}

	cout << "Correct predictions: " << correctPredictions << std::endl;
	double accuracy = static_cast<double>(correctPredictions) / testData.size();
	std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;


	return EXIT_SUCCESS;
}