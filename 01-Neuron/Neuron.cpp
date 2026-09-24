// Neuron.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
using namespace std;
 

double ReLu(double x) {
	return max(0.0, x);
}


class Neuron {
private:
	vector<double> weights;
	double bias;// shift the output

public:
	Neuron(const vector<double>& initialWeights, double initialBias) :
		weights(initialWeights), bias(initialBias) {
	}

	double forward(const vector<double>& inputs, double (*activitaionFunc)(double)) {
		if (inputs.size() != weights.size()) {
			throw invalid_argument("Input size must weight size.");
		}

		double weightedSum = 0.0;
		for (size_t i = 0; i < inputs.size(); i++) {
			weightedSum += inputs[i] * weights[i];
		}
		weightedSum += bias;

		return activitaionFunc(weightedSum);
	}
};


int main()
{
	vector<double> weights = { 0.5,-1.2,0.3 };
	double bias = 0.7;

	Neuron neuron(weights, bias);
	vector<double> inputs = { 1.0,0.5,-1.0 };
	double output = neuron.forward(inputs, ReLu);
	cout << "Output with ReLu " << output << endl;
	return 0;
}