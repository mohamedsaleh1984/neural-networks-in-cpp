#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <vector>
#include <map>

using namespace std;
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
    class Wine;

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
            filename(file), delimiter(del), header(head) {}
 
        /// <summary>
		/// Reads the CSV file and returns a vector of Wine objects.
        /// </summary>
        /// <returns></returns>
        vector<Wine> readCSV() {
            std::ifstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file: " + filename);
            }

            std::vector<Wine> wines;
            std::string line;
            if (header) {
                std::getline(file, line); // Skip header
            }
            while (std::getline(file, line)) {
                try {
                    Wine xwine(line);
                    wines.push_back(xwine);
                }
                catch (const std::exception& e) {
                    std::cerr << "Error parsing line: " << line << ". " << e.what() << std::endl;
                }
            }
            return wines;
        }

        /// <summary>
		/// Splits the dataset into training and testing sets based on the specified ratio.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="trainRatio"></param>
        /// <returns></returns>
        vector<vector<Wine>> splitData(const std::vector<Wine>& data, double trainRatio) {
            size_t trainSize = static_cast<size_t>(data.size() * trainRatio);
            std::vector<Wine> trainData(data.begin(), data.begin() + trainSize);
            std::vector<Wine> testData(data.begin() + trainSize, data.end());
            return {trainData, testData};
		}
    };

    class Wine {
    public:
        Wine() = default;
 
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
        int getLabel() const { return label; }
        const std::vector<double>& getAttributes() const { return attributes; }
    private:
        int label;
        std::vector<double> attributes;
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
            std::vector<std::pair<int,int>> distances;

            // Calculate distances from the test sample to all training samples 
            for (const auto& trainSample : trainData) {
                double dist = calculateDistance(trainSample.getAttributes(), testFeatures.getAttributes());
                distances.push_back({ dist, trainSample.getLabel()});
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
    
	// Read the wine dataset and split it into training and testing sets
    dataReader reader("wine.data", ",", false);
    vector<Wine> wines = reader.readCSV();

	// Split the dataset into training and testing sets
    vector<vector<Wine>> splitData = reader.splitData(wines, trainRatio);
    vector<Wine> trainData = splitData[0];
    vector<Wine> testData = splitData[1];

	// Evaluate the classifier on the test set
    Classifier classifier;
    int correctPredictions = 0;
    for (const auto& testSample : testData) {
        int predictedLabel = classifier.predictKNN(trainData, testSample);
        if (predictedLabel == testSample.getLabel()) {
            correctPredictions++;
        }
    }
    double accuracy = static_cast<double>(correctPredictions) / testData.size();
	std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;


	return EXIT_SUCCESS;
}