// mlp.cpp
//  A Hello World Example of Artificial Intelligence Using Multilayer Perceptron in C / C++
// https://www.youtube.com/watch?v=QvQB58TiiwI

#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

// MLP traning for an XOR gate
const int n1 = 3;			// number of inputs and bias
const int m1 = 3;			// number of hidden nodes and bias
const int K  = 1;			// number of outputs
const int numSamples = 4;
double inputs[numSamples][n1] = {

	1,0,0,					// x0=1, x1=0, x2=0		
	1,0,1,					// x0=1, x1=0, x2=1
	1,1,0,					// x0=1, x1=1, x2=0
	1,1,1					// x0=1, x1=1, x2=1	
};
double labels[numSamples][K] = { 0,1,1,0 }; // target output

// from input to hidden layer
double w[n1][m1] = {						// Random Weight
		0.97,	0.2,	0.7,			
		0.73,	0.1,	0.9,
		0.2,	0.7,	0.3
};
// from hidden layer to output
double wo[m1][K] = { 0.76, 0.6, 0.1 };		// Random weights


// Simple MLP class
class MLP {
private:
	double a[m1];	// linear sum 
	double h[m1];	// hiddden nodes 
	double y[K];	// predicted output
	double z[K];	// linear sum
	double eta;		// learning rate

public:
	MLP(double learing_rate) {
		eta = learing_rate;
		h[0] = 1;		// for bias
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
		
		return y;
	}


	// Backward propagation, yd is the desired output
	void backward(double yd[], double x[]) {

		double delta[m1];
		double deltao[m1];

		for (int l = 0; l < K; l++) {
			double e = yd[l] - y[l];			// output layer error
			deltao[l] = e * gd(z[l]);
		}

		// Compute hidden layer error
		for (int j = 0; j < m1; j++) {
			delta[j] = 0;
			for(int l = 0 ; l < K ; l++)
				delta[j] += deltao[l] * wo[j][l] * gd(a[j]);
		}

		// update weights (hidden to output)
		for (int j = 0; j < m1; j++)
			for (int l = 0; l < K; l++)
				wo[j][l] += eta * deltao[l] * h[j];

		// udpate weights (input to hidden)
		for (int i = 0; i < n1; i++)
			for (int j = 0; j < m1; j++)
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
				backward(yd + k * k, x);
			}
		}
	}

	void printWeigts() {

		cout << "\n Weights";
		for (int i = 0; i < n1; i++) {
			cout << endl;
			for (int j = 0; j < m1; j++) {
				printf(" w[%d,%d] %5.3f\t", i, j, w[i][j]);
			}
		}
			
		cout << "\n Weights";
		for (int i = 0; i < m1; i++)
		{
			cout << endl;
			for (int j = 0; j < K; j++)
			{
				printf(" wo[%d,%d] %5.3f\t", i, j, wo[i][j]);
			}
		}
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

	string gates = "   XOR   ";
	MLP mlp(0.5);

	mlp.train(4, 10000);

	// Test the MLP
	double input[3];
	cout << "\n Testing MLP " << endl;
	input[0] = 1;

	for (int i = 0; i < 4; i++) {
		input[1] = i & 1;
		input[2] = i >> 1;
		double* output = mlp.forward(input);
		cout << fixed << setprecision(0) << "  " << input[2]
			<< gates << input[1] << " = " << classifier(*output) 
			<< setprecision(2) << " ( " << output[0] << ") " << endl;
	}
	
	mlp.printWeigts();
	cout << endl << "Hello, AI World !" << endl;
	return 0;
}