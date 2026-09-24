// 03-FeedForward.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <chrono>
using namespace std;

/*
https://www.youtube.com/watch?v=FVgERZLoduE&list=PLgt-v47ixLfJDs0eSwEAzFjq0-mH0QOzN&index=4
*/
namespace Activitaion {
	inline double relu(double x) {
		return max(0.0, x);
	}
	inline double relu_dydx(double x) {
		return x > 0 ? 1.0 : 0.0;
	}
	inline double sigmoid(double x) {
		return 1.0 / (1.0 + exp(-x));
	}
	inline double sigmoid_dydx(double x) {
		double s = sigmoid(x);
		return s * (1.0 - s);
	}
}

class Matrix {
private:
	vector<vector<double>> data;
	size_t cols, rows;
public:
	Matrix(size_t rows, size_t cols) :rows(rows), cols(cols)
	{
		data.resize(rows, vector<double>(cols, 0.0));
	}
	double& operator()(size_t i, size_t j) {
		return data[i][j];
	}

	const double& operator()(size_t i, size_t j)const {
		return data[i][j];
	}

	size_t getRows()const {
		return rows;
	}

	size_t getCols()const {
		return cols;
	}
};

class NeuralNetwork {
private:
	// number of neurons in each layer
	vector<int> layerSizes;
	// weights
	Matrix weights1, weights2, weights3;
	vector<double> bias1, bias2, bias3;
	mt19937 gen;

	void initializeWeights() {
		normal_distribution<> dist(0.0, 1.0);
		double scale1 = sqrt(2.0 / layerSizes[0]);
		double scale2 = sqrt(2.0 / layerSizes[1]);
		double scale3 = sqrt(2.0 / layerSizes[2]);

		for (int i = 0; i < weights1.getRows(); i++) {
			for (int j = 0; j < weights1.getCols(); j++) {
				weights1(i, j) = dist(gen) * scale1;
			}
		}

		for (int i = 0; i < weights2.getRows(); i++) {
			for (int j = 0; j < weights2.getCols(); j++) {
				weights2(i, j) = dist(gen) * scale2;
			}
		}

		for (int i = 0; i < weights3.getRows(); i++) {
			for (int j = 0; j < weights3.getCols(); j++) {
				weights3(i, j) = dist(gen) * scale3;
			}
		}

		fill(bias1.begin(), bias1.end(), 0.0);
		fill(bias2.begin(), bias2.end(), 0.0);
		fill(bias3.begin(), bias3.end(), 0.0);
	}

public:
	NeuralNetwork(int inputSize, int hidden1Size, int hidden2Size, int outputSize) :
		layerSizes{ inputSize, hidden1Size,hidden2Size, outputSize },
		weights1(inputSize, hidden1Size),
		weights2(hidden1Size, hidden2Size),
		weights3(hidden2Size, outputSize),
		bias1(hidden1Size),
		bias2(hidden2Size),
		bias3(outputSize),
		gen(random_device{}())
	{
		if (inputSize <= 0 || hidden1Size <= 0 || hidden2Size <= 0 || outputSize <= 0)
		{
			throw invalid_argument("Layer sizes must be positive");
		}

		initializeWeights();
	}


	vector<double> forward(const vector<double>& input) {
		if (input.size() != layerSizes[0]) {
			throw runtime_error("Input size mismatch");
		}

		// First Layer = Input -> Hidden Layer 1  using Activitaion Function ReLu
		vector<double> hidden1(layerSizes[1]);
		for (int j = 0; j < layerSizes[1]; j++) {
			double sum = bias1[j];
			for (int i = 0; i < layerSizes[0]; i++) {
				sum += input[i] * weights1(i, j);
			}
			hidden1[j] = Activitaion::relu(sum);
		}

		// Second Layer = Hidden Layer 1 -> Hidden Layer 2 using Activitaion Function ReLu
		vector<double> hidden2(layerSizes[2]);
		for (int j = 0; j < layerSizes[2]; j++) {
			double sum = bias2[j];
			for (int i = 0; i < layerSizes[1]; i++) {
				sum += hidden1[i] * weights2(i, j);
			}
			hidden2[j] = Activitaion::relu(sum);
		}

		// Third Layer = Hidden Layer 2 -> Output Layer using Activitaion Function Sigmoid
		vector<double> output(layerSizes[3]);
		for (int j = 0; j < layerSizes[3]; j++) {
			double sum = bias3[j];
			for (int i = 0; i < layerSizes[2]; i++) {
				sum += hidden2[i] * weights3(i, j);
			}
			output[j] = Activitaion::sigmoid(sum);
		}

		return output;
	}

	void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, double learningRate, int epochs)
	{
		// inputs has corrosponding output
		if (inputs.size() != targets.size()) {
			throw runtime_error("input and target sizes don't match");
		}

		// Training Cycles
		for (int epoch = 0; epoch < epochs; epoch++) {
			double totalError = 0.0;
			for (size_t k = 0; k < inputs.size(); k++) {
				// Forward Pass: Compute output

				// Layer 1 from Input to Hidden Layer 1
				vector<double> hidden1(layerSizes[1]);
				vector<double> hidden1Pre(layerSizes[1]);
				// Layer 1.
				for (int j = 0; j < layerSizes[1]; j++) {
					double sum = bias1[j];
					for (int i = 0; i < layerSizes[0]; i++) {
						sum += inputs[k][i] * weights1(i, j);
					}
					hidden1Pre[j] = sum;
					hidden1[j] = Activitaion::relu(sum);
				}

				// Layer 2
				vector<double> hidden2(layerSizes[2]);
				vector<double> hidden2Pre(layerSizes[2]);
				for (int j = 0; j < layerSizes[2]; j++) {
					double sum = bias2[j];
					for (int i = 0; i < layerSizes[1]; i++) {
						sum += hidden1[i] * weights2(i, j);
					}
					hidden2Pre[j] = sum;
					hidden2[j] = Activitaion::relu(sum);
				}

				// Output Layer
				vector<double> output(layerSizes[3]);
				vector<double> outputPre(layerSizes[3]);
				for (int j = 0; j < layerSizes[3]; j++) {
					double sum = bias3[j];
					for (int i = 0; i < layerSizes[2]; i++) {
						sum += hidden2[i] * weights3(i, j);
					}
					outputPre[j] = sum;
					output[j] = Activitaion::sigmoid(sum);
				}


				// Loss
				for (int j = 0; j < layerSizes[3]; j++) {
					double error = targets[k][j] - output[j];
					totalError += error * error;
				}

				vector<double> outputGradiants(layerSizes[3]);
				for (int j = 0; j < layerSizes[3]; j++) {
					outputGradiants[j] = (output[j] - targets[k][j]) * Activitaion::sigmoid_dydx(outputPre[j]);
				}

				vector<double> hidden2Gradiants(layerSizes[2]);
				for (int i = 0; i < layerSizes[2]; i++) {
					double error = 0;
					for (int j = 0; j < layerSizes[3]; j++) {
						error += outputGradiants[j] * weights3(i, j);
					}
					hidden2Gradiants[i] = error * Activitaion::relu_dydx(hidden2Pre[i]);
				}

				vector<double> hidden1Gradiants(layerSizes[1]);
				for (int i = 0; i < layerSizes[1]; i++) {
					double error = 0;
					for (int j = 0; j < layerSizes[2]; j++) {
						error += hidden2Gradiants[j] * weights2(i, j);
					}
					hidden1Gradiants[i] = error * Activitaion::relu_dydx(hidden1Pre[i]);
				}


				//---------------------------- Update Weights/Bias for Layer 3 ------------------------
				// Update Weights for Layer 3
				for (int i = 0; i < layerSizes[2]; i++) {
					for (int j = 0; j < layerSizes[3]; j++) {
						weights3(i, j) -= learningRate * outputGradiants[j] * hidden2[i];
					}
				}

				// Update Bias for Layer 3
				for (int j = 0; j < layerSizes[3]; j++) {
					bias3[j] -= learningRate * outputGradiants[j];
				}

				//---------------------------- Update Weights/Bias for Layer 2 ------------------------
				// Update Weight for Layer 2
				for (int i = 0; i < layerSizes[1]; i++) {
					for (int j = 0; j < layerSizes[2]; j++)
					{
						weights2(i, j) -= learningRate * hidden2Gradiants[j] * hidden1[i];
					}
				}

				// Update Bias for Layer 2
				for (int j = 0; j < layerSizes[2]; j++) {
					bias2[j] -= learningRate * hidden2Gradiants[j];
				}

				//---------------------------- Update Weights/Bias for Layer 1 ------------------------
				// Update weights for Layer 1
				for (int i = 0; i < layerSizes[0]; i++) {
					for (int j = 0; j < layerSizes[1]; j++) {
						weights1(i, j) -= learningRate * hidden1Gradiants[j] * inputs[k][i];
					}
				}

				// Update Bias for Layer 2
				for (int j = 0; j < layerSizes[1]; j++) {
					bias1[j] -= learningRate * hidden1Gradiants[j];
				}
			}

			if (epoch % 100 == 0) {
				cout << "EPOCH : " << epoch << "   MSE : " << totalError / inputs.size() << "\n";
			}

		}
	}
};

int main()
{
	try
	{
		// 2 Inputs
		// 8 Neurons 
		// 4 Hidden Layers
		// 1 Outout
		NeuralNetwork nn(2, 8, 4, 1);
		mt19937 gen(random_device{}());
		uniform_real_distribution<>  dist(-2.0, 2.0);

		const int numSamples = 1000;
		// Input 
		vector<vector<double>> inputs(numSamples);
		// Expected Outcome
		vector<vector<double>> targets(numSamples);

		for (int i = 0; i < numSamples; i++) {
			// Generate Randome Points
			double x = dist(gen);
			double y = dist(gen);
			// Add them to input vector
			inputs[i] = { x,y };

			// check the distance 
			double distance = sqrt(x * x + y * y);
			targets[i] = { distance < 1.0 ? 1.0 : 0.0 };
		}
		auto  start = chrono::high_resolution_clock::now();
		nn.train(inputs, targets, 0.01, 1000);
		auto end = chrono::high_resolution_clock::now();
		cout << "Total Training Time " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms\n";

		vector<vector<double>> testPoints = {
			{0.0,0.0},
			{1.0,1.0},
			{0.5,0.5},
			{2.0,0.0},
		};

		cout << "\n Test Results (1 = inside, 0 = outside) : " << endl;

		for (const auto& point : testPoints) {
			auto output = nn.forward(point);
			double actual = sqrt(point[0] * point[0] + point[1] * point[1]) < 1.0 ? 1.0 : 0.0;
			cout << "  point( " << point[0] << ", " << point[1] << " ) :"
				<< output[0] << " ( actual : " << actual
				<< " ,  error " << abs(output[0] - actual) << " ) \n";
		}

	}
	catch (const std::exception& e)
	{
		cout << "ERROR : " << e.what() << endl;
	}

	return 0;
}
