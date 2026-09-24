// NeuralNetworksInCpp.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
using namespace std;

// Define a type shortcut for Loss Function
using LossFunctionCallback = double(*)(vector<double>&, vector<double>&);


// Activitaion Functions
// ReLu - Rectified Linear Unit
static double relu(double x) {
	return (x > 0) ? x : 0;
}

static double relu_dydx(double x) {
	return (x > 0) ? 1 : 0;
}
/**
 * @brief A mathematical curve shaped like the letter S that maps any real number into a value between 0 and 1
 *
 * Acts as an activation function in machine learning and logistic regression to introduce non-linearity.
 * https://www.storyofmathematics.com/derivative-of-sigmoid/
 * @param x
 * @return A real number between 0 and 1 AND 0.5 IF x is exactly 0
 */
static double sigmoid(double z) {
	return 1 / (1 + exp(-z));
}

/**
* @brief Use the quotient rule or the chain rule to calculate the derivative
* plays an important role in changing weights during backpropagation in neural network training and has
* different characteristcis Symmetry, Monotonicity and Range between 0 and 0.25
* https://www.storyofmathematics.com/derivative-of-sigmoid/
* @return A real number between 0 and 0.25
*/
static double sigmoid_dydx(double z) {
	return z * (1 - z);
}

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

	// Forward Propagation
	vector<double> forward(vector<double> input, vector<double>& hiddenLayer)
	{
		hiddenLayer.assign(weights1[0].size(), 0);
		for (size_t i = 0; i < weights1.size(); i++) {
			for (size_t j = 0; j < weights1[0].size(); j++)
			{
				hiddenLayer[j] += input[i] * weights1[i][j];
			}
		}
		for (size_t j = 0; j < hiddenLayer.size(); j++) {
			hiddenLayer[j] = relu(hiddenLayer[j] + bias1[j]);
		}

		vector<double> output(weights2[0].size(), 0);
		for (size_t i = 0; i < weights2.size(); i++) {
			for (size_t j = 0; j < weights2[0].size(); j++)
			{
				output[j] += hiddenLayer[i] * weights2[i][j];
			}
		}

		for (size_t j = 0; j < output.size(); j++)
			output[j] = sigmoid(output[j] + bias2[j]);

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
					outputGradiants[j] = (output[j] - targets[i][j]) * sigmoid_dydx(output[j]);
				}

				vector<double> hiddenGradients(hiddenLayer.size());
				for (size_t j = 0; j < hiddenLayer.size(); j++) {
					hiddenGradients[j] = 0;
					for (size_t k = 0; k < output.size(); k++) {
						hiddenGradients[j] += outputGradiants[k] * weights2[j][k];
					}

					hiddenGradients[j] *= relu_dydx(hiddenLayer[j]);
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


int main() {
	NeuralNetwork nn(2, 3, 1);

	// XOR Traning data
	vector<vector<double>> inputs = { {0,0},{0,1},{1,0},{1,1} };	// 4x2
	vector<vector<double>> targets = { {0},{1},{1},{0} };			//4x1

	nn.train(inputs, targets, 0.05, 3000, meanAbsoluteError);

	cout << "\nTraining Complete\n";
	cout << "==================\n";
	cout << "Input               Target    Raw     Predicted    Result\n";
	cout << "---------------------------------------------------------\n";
	int correct = 0;
	for (size_t i = 0; i < inputs.size(); i++) {
		double raw = nn.predict(inputs[i])[0];
		int pred = raw >= 0.5 ? 1 : 0;
		int target = (int)targets[i][0];
		bool ok = pred == target;
		if (ok) correct++;
		cout << "[" << fixed << setprecision(4) << inputs[i][0] << "," << setprecision(4) << inputs[i][1] << "]"
			<< "      " << target
			<< "       " << fixed << setprecision(4) << raw
			<< "     " << pred
			<< "          " << (ok ? "PASS" : "FAIL") << "\n";
	}
	cout << "---------------------------------------------------------\n";
	cout << "Accuracy: " << correct << "/" << inputs.size() << "\n";
	cout << defaultfloat;   // reset formatting

	cout << "\n Testing Phase \n" << endl;
	vector<vector<double>> test_inputs = { {1,1},{1,0},{0,0},{1,1},{0,1},{1,1},{0,1},{0,0},{0,1},{1,0},{1,1} };

	for (size_t i = 0; i < test_inputs.size(); i++)
	{
		vector<double> prediction = nn.predict(test_inputs[i]);
		int rounded = (prediction[0] >= 0.5) ? 1 : 0;

		cout << " test_input : [ " << fixed << setprecision(2) << test_inputs[i][0] << " , "
			<< fixed << setprecision(2) << test_inputs[i][1] << "] -> prediction : "
			<< fixed << setprecision(2) << prediction[0] << " | -> raw: " << rounded << endl;
	}

	return 0;
}