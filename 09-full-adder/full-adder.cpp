// mlp.cpp
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

using namespace std;

// MLP traning for an XOR gate
// 3 bits + Carry bit
const int n1 = 4;			// number of inputs and bias
const int m1 = 6;			// number of hidden nodes and bias
const int K = 2;			// number of outputs  ( S  C )
const int numSamples = 8;	// 
double inputs[numSamples][n1] = {
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
double labels[numSamples][K] = {
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
double w[n1][m1] = {
	0.97, 0.2, 0.7, 0.5, 0.3,
	0.73, 0.1, 0.9, 0.4, 0.6,
	0.2,  0.7, 0.3, 0.8, 0.1,
	0.5,  0.6, 0.2, 0.9, 0.4
};

// from hidden layer to output
double wo[m1][K] = { 0.76, 0.3,
	0.6,  0.7,
	0.1,  0.5,
	0.4,  0.8,
	0.9,  0.2 };


// Simple MLP class
class MLP {
private:
	double a[m1];	// linear sum of products of inputs and weights
	double h[m1];	// hiddden nodes 
	double y[K];	// predicted output
	double z[K];	// linear sum
	double eta;		// learning rate

	void initWeights() {	
		std::random_device rd;
		mt19937 gen(rd());
		normal_distribution<> dist(std::nextafter(0.0, 1.0), 1.0);

		double scale1 = sqrt(2.0 / n1);
		double scale2 = sqrt(2.0 / m1);

		for (int i = 0; i < n1; i++)
			for (int j = 0; j < m1; j++)
				w[i][j] =fabs( dist(gen));

		for (int i = 0; i < m1; i++)
			for (int j = 0; j < K; j++)
				wo[i][j] =fabs( dist(gen));
	}

public:
	MLP(double learing_rate) {
		eta = learing_rate;
		h[0] = 1;		// for bias
		//initWeights();
	}

	// Sigmoid activitaion function
	double g(double x) {
		return 1.0 / (1.0 + exp(-x));
	}

	// Derivayive of sigmod function
	double gd(double x) {
		double s = g(x);
		return s * (1 - s);
	}

	double* forward(double x[]) {
		// hidden layer activitaion
		for (int j = 0; j < m1; j++) {
			a[j] = 0;
			for (int i = 0; i < n1; i++)
				a[j] += w[i][j] * x[i];
			if (j > 0)
				h[j] = g(a[j]);
		}

		// output layer activitaion
		for (int l = 0; l < K; l++) {
			z[l] = 0;
			for (int j = 0; j < m1; j++)
				z[l] += wo[j][l] * h[j];
			y[l] = g(z[l]);
		}

		return y;		// return output
	}


	// Backward propagation, yd is the desired output
	void backward(double yd[], double x[]) {

		double delta[m1];
		double deltao[K];

		for (int l = 0; l < K; l++) {
			double e = yd[l] - y[l];			// output layer error
			deltao[l] = e * gd(z[l]);
		}

		// Compute hidden layer error
		for (int j = 1; j < m1; j++) {
			delta[j] = 0;
			for (int l = 0; l < K; l++)
				delta[j] += deltao[l] * wo[j][l] * gd(a[j]);
		}

		// update weights (hidden to output)
		for (int j = 0; j < m1; j++)
			for (int l = 0; l < K; l++)
				wo[j][l] += eta * deltao[l] * h[j];

		// udpate weights (input to hidden)
		for (int i = 0; i < n1; i++)
			for (int j = 1; j < m1; j++)
				w[i][j] += eta * delta[j] * x[i];

	}

	void get_input_data(int k, double x[]) {
		for (int i = 0; i < n1; i++)
			x[i] = inputs[k][i];
	}

	// Train the MLP
	// epochs = number of times of training
	// numSamples = number of different inputs sets
	void train(int numSamples, int epochs) {
		double x[n1];			// inputs
		double* yd = &labels[0][0];

		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int k = 0; k < numSamples; k++)
			{
				get_input_data(k, x);
				forward(x);
				backward(yd + k * K, x);
			}
		}
	}

	void printWeigts() {

		cout << "\nMLP input to hidden weights";
		for (int i = 0; i < n1; i++) {
			cout << endl;
			for (int j = 0; j < m1; j++)
			{
				printf("w[%d][%d] %5.3f\t", i, j, w[i][j]);
			}
		}

		cout << endl;

		cout << "\nMLP hidden to output weights";
		for (int i = 0; i < m1; i++)
		{
			cout << endl;
			for (int j = 0; j < K; j++)
			{
				printf("wo[%d][%d] %5.3f\t", i, j, wo[i][j]);
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

	double input[n1];
	cout << "\n Testing MLP (Full Adder) " << endl;
	
	for (int i = 0; i < numSamples; i++) {
		input[1] = i & 1;            // A
		input[2] = (i >> 1) & 1;     // B
		input[3] = (i >> 2) & 1;     // Cin

		double* output = mlp.forward(input);

		cout << fixed << setprecision(0)
			<< "  " << input[3] << gates << input[2] << gates << input[1]
			<< " = Sum:" << classifier(output[0])
			<< " Cout:" << classifier(output[1])
			<< setprecision(2)
			<< " (" << output[0] << ", " << output[1] << ")" << endl;
	}

	mlp.printWeigts();
	cout << endl << "Hello, AI World !" << endl;
	return 0;
}