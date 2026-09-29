// full-adder.cpp
// A Hello World Example of Artificial Intelligence Using Multilayer Perceptron in C / C++
// https://www.youtube.com/watch?v=QvQB58TiiwI

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <chrono>
using namespace std;

// MLP traning for an Full Adder gate
// 1 Bias	2 bits Input 1 Carry bit
const int nn_inputs_count = 4;				// number of inputs and bias
const int nn_hiddenLayerNodes_count = 6;	// number of hidden nodes and bias
const int nn_output_count = 2;				// number of outputs  ( S  C )
const int numSamples = 8;

// 1 Bias	2 bits Input 1 Carry bit
// First row is always 1 
double inputs[numSamples][nn_inputs_count] = {
	1, 0, 0, 0,   // A=0 B=0 Cin=0
	1, 1, 0, 0,   // A=0 B=0 Cin=1
	1, 0, 1, 0,   // A=0 B=1 Cin=0
	1, 1, 1, 0,   // A=0 B=1 Cin=1

	1, 0, 0, 1,   // A=1 B=0 Cin=0
	1, 1, 0, 1,   // A=1 B=0 Cin=1
	1, 0, 1, 1,   // A=1 B=1 Cin=0
	1, 1, 1, 1    // A=1 B=1 Cin=1				
};

// target output
double nn_expected_output[numSamples][nn_output_count] = {
	0,0,   // 0+0+0 = 0, carry 0
	1,0,   // 0+0+1 = 1, carry 0
	1,0,   // 0+1+0 = 1, carry 0
	0,1,   // 0+1+1 = 0, carry 1
	1,0,   // 1+0+0 = 1, carry 0
	0,1,   // 1+0+1 = 0, carry 1
	0,1,   // 1+1+0 = 0, carry 1
	1,1    // 1+1+1 = 1, carry 1
}; 

// from input to hidden layer
double weight[nn_inputs_count][nn_hiddenLayerNodes_count] = {
	0.97, 0.2, 0.7, 0.5,0.3, 0.73,0.1, 0.9,
	0.4, 0.6,0.2, 0.7,0.3, 0.8,0.1, 0.5,0.6, 0.2,0.9, 0.4
};

// from hidden layer to output
double weightOutput[nn_hiddenLayerNodes_count][nn_output_count] = 
{ 0.76,  0.3,0.6,  0.7,0.1,  0.5,0.4,  0.8,0.9,  0.2 };


// Simple MLP class
class MLP {
private:
	double wigIn2Hid[nn_hiddenLayerNodes_count];	// linear sum of products of inputs and weights
	double actIn2Hid[nn_hiddenLayerNodes_count];	// hiddden nodes 
	double actHid2Out[nn_output_count];				// predicted output
	double wigHid2Out[nn_output_count];				// linear sum
	double eta;										// learning rate

public:
	MLP(double learing_rate) {
		eta = learing_rate;
		actIn2Hid[0] = 1;		// for bias
	}

	// Sigmoid activitaion function
	double sigmoid(double x) {
		return 1.0 / (1.0 + exp(-x));
	}

	// Derivayive of sigmod function
	double sigmoid_dydx(double x) {
		double s = sigmoid(x);
		return s * (1 - s);
	}

	double* forward(double input[]) {
		// hidden layer activation
		for (int j = 0; j < nn_hiddenLayerNodes_count; j++) {
			wigIn2Hid[j] = 0;
			for (int i = 0; i < nn_inputs_count; i++)
				wigIn2Hid[j] += weight[i][j] * input[i];
			// Compute Activitaion After input*weight
			actIn2Hid[j] = sigmoid(wigIn2Hid[j]);
		}

		
		// output layer activitaion
		for (int l = 0; l < nn_output_count; l++) {
			wigHid2Out[l] = 0;
			for (int j = 0; j < nn_hiddenLayerNodes_count; j++)
				wigHid2Out[l] += weightOutput[j][l] * actIn2Hid[j];
			actHid2Out[l] = sigmoid(wigHid2Out[l]);
		}
		
		return actHid2Out;		// return output
	}

	// Backward propagation, expected_output is the desired output
	void backward(double expected_output[], double x[]) {

		double delta[nn_hiddenLayerNodes_count];
		double deltao[nn_output_count];
		
		// Computer Delta
		for (int l = 0; l < nn_output_count; l++) {
			double e = expected_output[l] - actHid2Out[l];			// output layer error
			deltao[l] = e * sigmoid_dydx(wigHid2Out[l]);
		}

		// Compute hidden layer error
		for (int j = 0; j < nn_hiddenLayerNodes_count; j++) {
			delta[j] = 0;
			for (int l = 0; l < nn_output_count; l++)
				delta[j] += deltao[l] * weightOutput[j][l] * sigmoid_dydx(wigIn2Hid[j]);
		}

		// update weights (hidden to output)
		for (int j = 0; j < nn_hiddenLayerNodes_count; j++)
			for (int l = 0; l < nn_output_count; l++)
				weightOutput[j][l] += eta * deltao[l] * actIn2Hid[j];

		// udpate weights (input to hidden)
		for (int i = 0; i < nn_inputs_count; i++)
			for (int j = 0; j < nn_hiddenLayerNodes_count; j++)
				weight[i][j] += eta * delta[j] * x[i];

	}

	void get_input_data(int k, double x[]) {
		for (int i = 0; i < nn_inputs_count; i++)
			x[i] = inputs[k][i];

		// printArray("inputs", x, nn_inputs_count);
	}

	// Train the MLP
	// epochs = number of times of training
	// numSamples = number of different inputs sets
	void train(int numSamples, int epochs) {
		double x[nn_inputs_count];							// inputs
		double* expected_output = &nn_expected_output[0][0];

		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int k = 0; k < numSamples; k++)
			{
				get_input_data(k, x);
				forward(x);
				backward(expected_output + k * nn_output_count, x);
			}

			/*if (epoch % 100 == 0) {
				printArray("inputs", x, nn_inputs_count);
				printWeigts();
			}*/
		}
	}

	void printArray(string header,double x[], int len) {
		cout << header << " " << endl;
		for (int i = 0; i < len; i++) {
			cout << x[i] << " ";
		}
		cout << endl;
	}
	void printWeigts() {

		cout << "\nMLP input to hidden weights";
		for (int i = 0; i < nn_inputs_count; i++) {
			cout << endl;
			for (int j = 0; j < nn_hiddenLayerNodes_count; j++)
			{
				printf("weight[%d][%d] %5.3f\t", i, j, weight[i][j]);
			}
		}

		cout << endl;

		cout << "\nMLP hidden to output weights";
		for (int i = 0; i < nn_hiddenLayerNodes_count; i++)
		{
			cout << endl;
			for (int j = 0; j < nn_output_count; j++)
			{
				printf("weightOutput[%d][%d] %5.3f\t", i, j, weightOutput[i][j]);
			}
		}

		cout << endl;
	}

	~MLP() {

	}
};

int classifier(double x) {
	if (x< 0.2 && x > -0.1)
		return 0;

	if (x > 0.8 && x < 1.1)
		return 1;

	return -1;
}

int main() {

	// Training data for XOR gate, four sets of x
	string gates = " + ";
	MLP mlp(0.5);

	mlp.train(numSamples, 20000);

	double input[nn_inputs_count];
	cout << "\n Testing MLP (Full Adder) " << endl;
	
	for (int i = 0; i < numSamples; i++) {
		input[1] = i & 1;            // A
		input[2] = (i >> 1) & 1;     // B
		input[3] = (i >> 2) & 1;     // Cin

		cout << input[1] << " + " << input[2] << " + " << input[3] << endl;
		double* output = mlp.forward(input);

		cout << fixed << setprecision(0)
			<< "  " << input[1] 
			<< gates << input[2] 
			<< gates << input[3]
			<< " = Sum:" << classifier(output[0])
			<< " Cout:" << classifier(output[1])
			<< setprecision(2)
			<< " (" << output[0] 
			<< ", " << output[1] 
			<< ")" << endl;
	}

	mlp.printWeigts();
	cout << endl << "Hello, AI World !" << endl;
	return 0;
}