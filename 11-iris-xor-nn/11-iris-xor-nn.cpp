// 11.iris-xor-nn.cpp
// 
#pragma region headers

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


#pragma endregion

// Define a type shortcut for Loss Function
using LossFunctionCallback = double(*)(vector<double>&, vector<double>&);

namespace activition_functions {

	// tanh(x) = (exp(x) - exp(-x))/ (exp(x) + exp(-x))
	double tanh(double x) {
		return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	}

	// tanh'(x) = 1 - (tanh(x))^2)
	double tanh_dydx(double x) {
		double th = tanh(x);
		return 1 - pow(th, 2);
	}

	// f(x_i) = e^(x_i) / sum_j e^(x_j)   (numerically stable)
	std::vector<double> softmax(const std::vector<double>& inputs) {
		if (inputs.empty())
		{
			throw runtime_error("inputs for softmax functons should not be empty.");
		}

		// 0. Copy logits
		vector<double> logits;
		logits.resize(inputs.size());
		for (int i = 0; i < inputs.size(); i++) {
			logits[i] = inputs[i];
		}

		// 1. Find the maximum value to avoid overflow
		double max_logit = *std::max_element(logits.begin(), logits.end());

		// 2. Compute exponentials and their sum simultaneously
		double sum = 0.0;
		for (auto& val : logits) {
			val = std::exp(val - max_logit);
			sum += val;
		}

		// 3. Divide each element by the sum to get probabilities
		for (auto& val : logits) {
			val /= sum;
		}

		return logits;
	}

	// Jacobian: d f_i / d x_j = f_i * (delta_ij - f_j)
	// Returns full Jacobian matrix (n x n)
	std::vector<std::vector<double>> softmax_dydx(const std::vector<double>& x) {
		std::vector<double> f = softmax(x);
		size_t n = f.size();
		std::vector<std::vector<double>> J(n, std::vector<double>(n, 0.0));

		for (size_t i = 0; i < n; ++i) {
			for (size_t j = 0; j < n; ++j) {
				double delta = (i == j) ? 1.0 : 0.0;
				J[i][j] = f[i] * (delta - f[j]);
			}
		}
		return J;
	}
}

namespace loss_function {
	// LOSS Functions : a mathematical method used to measure how wrong a machine learning model's predictions are compared to the true target values.
	/**
	@brief Measures the average of the squares of the errors in another words
	// https://www.geeksforgeeks.org/machine-learning/ml-common-loss-functions/

	@return
	*/
	double meanSquaredError(vector<double>& predictions, vector<double>& targets) {
		double sum = 0;
		for (size_t i = 0; i < predictions.size(); i++)
		{
			sum += (predictions[i] - targets[i]) * (predictions[i] - targets[i]);
		}
		// how far our predictions far from the actual values.
		return sum / predictions.size();
	}

	/**
	* @brief is a measure of errors between paired observations expressing the same phenomenon.
	* https://en.wikipedia.org/wiki/Mean_absolute_error
	*/
	static double meanAbsoluteError(vector<double>& predictions, vector<double>& targets) {
		double sum = 0;
		for (size_t i = 0; i < predictions.size(); i++)
		{
			sum += fabs((predictions[i] - targets[i]));
		}
		// how far our predictions far from the actual values.
		return sum / predictions.size();
	}

	/**
	* s@brief help in determining whether the model has a positive bias or negative bias.s
	*/
	static double meanBiasError(vector<double>& predictions, vector<double>& targets) {
		double sum = 0;
		for (size_t i = 0; i < predictions.size(); i++)
		{
			sum += (predictions[i] - targets[i]);
		}
		// how far our predictions far from the actual values.
		return sum / predictions.size();
	}

}

namespace random_function {
	class RandomGenerator {
	public:
		RandomGenerator(double lb = -0.1, double up = 0.1) :lowerBound(lb), upperBound(up) {
		}

		double pick() {
			mt19937 gen(random_device{}());
			uniform_real_distribution<>  dist(lowerBound, upperBound);

			return dist(gen);
		}
	private:
		double lowerBound;
		double upperBound;
	};
}

namespace nn {
	using namespace loss_function;

	class NeuralNetwork {
	public:
		vector<vector<double>> weights1, weights2;
		vector<double> bias1, bias2;

		NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
			weights1.resize(inputSize, vector<double>(hiddenSize));
			weights2.resize(hiddenSize, vector<double>(outputSize));

			// start with small positive value
			bias1.resize(hiddenSize, 0.1);
			bias2.resize(outputSize, 0.1);

			//
			initWeights();
		}

		void initWeights() {

			for (auto& row : weights1) {
				for (double& w : row)
				{
					// Generate random values
					// 0.1 - 0.5
					w = (double)rand() / RAND_MAX - 0.5;
				}
			}

			for (auto& row : weights2) {
				for (double& w : row)
				{
					// Generate random values
					// 0.1 - 0.5
					w = (double)rand() / RAND_MAX - 0.5;
				}
			}
		}

		// Forward Pass
		vector<double> forward(vector<double> input, vector<double>& hiddenLayer)
		{
			hiddenLayer.assign(weights1[0].size(), 0);
			for (size_t i = 0; i < weights1.size(); i++) {
				for (size_t j = 0; j < weights1[0].size(); j++)
				{
					hiddenLayer[j] += input[i] * weights1[i][j];
				}
			}

			// From Input Layer to Hidden Layer
			for (size_t j = 0; j < hiddenLayer.size(); j++) {
				// softmax..			
				// hiddenLayer[j] = relu(hiddenLayer[j] + bias1[j]);
			}

			vector<double> output(weights2[0].size(), 0);
			for (size_t i = 0; i < weights2.size(); i++) {
				for (size_t j = 0; j < weights2[0].size(); j++)
				{
					output[j] += hiddenLayer[i] * weights2[i][j];
				}
			}

			// From Hidden Layer to Output Layer
			for (size_t j = 0; j < output.size(); j++) {
				// output
				// output[j] = sigmoid(output[j] + bias2[j]);
			}
				
			return output;
		}

		vector<double> predict(vector<double> input) {
			vector<double> hiddenLayer;
			return forward(input, hiddenLayer);
		}

		// Training With Backpropagagtion
		/*
		// @input  => xor table
		// @output => expected values for each input
		// @learningRate => Raio of changing the bias values
		// @epochs => How many Iterations the model will go over the entire dataset
		// @LossFunctionCallback => If not passed then MeanSquareError otherwise Custom Implementation
		*/
		void train(vector<vector<double>> inputs, vector<vector<double>> targets, double learningRate, int epochs, LossFunctionCallback LossFunction = nullptr) {
			for (int e = 0; e < epochs; e++) {
				// keep track of the total errors across the training sample
				// judge how well the network doing over time.
				double totalLoss = 0;

				for (size_t i = 0; i < inputs.size(); i++) {
					vector<double> hiddenLayer;
					vector<double> output = forward(inputs[i], hiddenLayer);

					// Factoring out the Loss Function to Check Loss Function impact
					double loss = LossFunction == nullptr ? meanSquaredError(output, targets[i]) : LossFunction(output, targets[i]);
					totalLoss += loss;

					// Compute  Stochastic Gradient Descent  (Backpropagation)
					//  because the weights are updated after each individual training sample, not after the entire dataset. 
					// This is a common and efficient variant of gradient descent, especially for small datasets like XOR.

					vector<double> outputGradiants(output.size());
					for (size_t j = 0; j < output.size(); j++)
					{
						// CORRECT
						//outputGradiants[j] = (output[j] - targets[i][j]) * sigmoid_dydx(output[j]);
					}

					vector<double> hiddenGradients(hiddenLayer.size());
					for (size_t j = 0; j < hiddenLayer.size(); j++) {
						hiddenGradients[j] = 0;
						for (size_t k = 0; k < output.size(); k++) {
							hiddenGradients[j] += outputGradiants[k] * weights2[j][k];
						}
						// Correct
						//hiddenGradients[j] *= relu_dydx(hiddenLayer[j]);
					}


					// Update Weights and Biases
					for (size_t j = 0; j < weights2.size(); j++)
					{
						for (size_t k = 0; k < weights2[0].size(); k++) {
							weights2[j][k] -= learningRate * outputGradiants[k] * hiddenLayer[j];
						}
					}

					for (size_t j = 0; j < bias2.size(); j++) {
						bias2[j] -= learningRate * outputGradiants[j];
					}

					for (size_t j = 0; j < weights1.size(); j++) {
						for (size_t k = 0; k < weights1[0].size(); k++) {
							weights1[j][k] -= learningRate * hiddenGradients[k] * inputs[i][j];
						}
					}

					for (size_t j = 0; j < bias1.size(); j++) {
						bias1[j] -= learningRate * hiddenGradients[j];
					}
				}
				if (e % 1000 == 0) {
					cout << "EPOCH : " << e << " LOSS : " << totalLoss / inputs.size() << endl;
				}
			}
		}

	};
}

namespace data_layer {
	/// <summary>
	/// iris entity
	/// </summary>
	struct iris {
		std::vector<double> features;	// [sepal_length, sepal_width, petal_length, petal_width]
		std::string label;				// Species name
		double expected;				// output flag
	};

	/// <summary>
	/// iris class data range
	/// </summary>
	vector<vector<int>>  data_range = {
		{0,49},{50,99},{100,149}
	};

	vector<iris> getDataset() {
		std::vector<iris> dataset;
		std::ifstream file("iris.csv");

		if (!file.is_open()) {
			std::cerr << "Error: Could not open iris.csv file." << std::endl;
			return dataset;
		}

		std::string line;

		// Load data from CSV
		while (std::getline(file, line)) {
			if (line.empty()) continue;

			std::stringstream ss(line);
			std::string token;
			iris sample;

			for (int i = 0; i < 4; ++i) {
				std::getline(ss, token, ',');
				sample.features.push_back(std::stod(token));
			}
			std::getline(ss, token, ',');
			sample.label = token;

			// Set Expected Flag for each class 
			if (token == "Iris-setosa") {
				sample.expected = 0;
			}
			else if (token == "Iris-versicolor") {
				sample.expected = 1;
			}
			else {
				sample.expected = 2;
			}

			dataset.push_back(sample);
		}

		file.close();

		return dataset;
	}

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

	vector<vector<iris>> getTraningTestData() {

		vector<iris> dataset = getDataset();

		vector<vector<iris>> ret;

		vector<iris> t = get_training_dataset(dataset);
		vector<iris> tes = get_test_dataset(dataset);

		random_shuffle(t.begin(), t.end());
		random_shuffle(tes.begin(), tes.end());

		ret.push_back(t);
		ret.push_back(tes);

		return ret;
	}
}


namespace demo_softmax {
	// Computes the stable softmax of an input vector in place
	void softmax(std::vector<double>& logits) {
		if (logits.empty()) return;

		// 1. Find the maximum value to avoid overflow
		double max_logit = *std::max_element(logits.begin(), logits.end());

		// 2. Compute exponentials and their sum simultaneously
		double sum = 0.0;
		for (auto& val : logits) {
			val = std::exp(val - max_logit);
			sum += val;
		}

		// 3. Divide each element by the sum to get probabilities
		for (auto& val : logits) {
			val /= sum;
		}
	}

	void demo() {
		// Example logits (raw model outputs)
		std::vector<double> inputs = { 2.0, 1.0, 0.1, 100.0 };

		std::cout << "Original inputs (including a large value): \n";
		for (double x : inputs) std::cout << x << " ";
		std::cout << "\n\n";

		// Apply Softmax
		softmax(inputs);

		std::cout << "Softmax Probabilities: \n";
		for (double p : inputs) {
			std::cout << p << " ";
		}
		std::cout << "\n";
	}

	void demo2() {
		// Class names corresponding to index 0, 1, and 2
		std::vector<std::string> classes = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };

		// Example raw output (logits) from a neural network's final layer for one sample
		std::vector<double> mock_logits = { 2.5, 1.1, -0.8 };

		// Calculate probabilities
		softmax(mock_logits);

		// Find the class with the highest probability (prediction)
		size_t predicted_class_idx = std::distance(
			mock_logits.begin(),
			std::max_element(mock_logits.begin(), mock_logits.end())
		);

		// Print Results
		std::cout << std::fixed << std::setprecision(4);
		std::cout << "--- Softmax Output for Iris Sample ---\n";
		for (size_t i = 0; i < classes.size(); ++i) {
			std::cout << classes[i] << ": Logit = " << mock_logits[i]
				<< " -> Probability = " << mock_logits[i] * 100 << "%\n";
		}

	}
}
int main()
{
	// 
	//nn::NeuralNetwork nx(4, 5, 3);

	demo_softmax::demo2();

	return EXIT_SUCCESS;
}
