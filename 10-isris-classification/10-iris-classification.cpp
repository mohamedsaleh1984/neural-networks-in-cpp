#include <iostream>
#include <fstream>
#include <string>
#include <exception>
#include "abstraction.cpp"
using namespace std;

#define filePath "iris.csv"
const int irisSize = 150;

// Constructor
Perceptron::Perceptron(int inputSize, ActivationFunction* activationFunction, char label) {
	this->activationFunction = activationFunction;
	this->label = label;
	this->delta = 0;
	this->weights = (double*)malloc(inputSize * sizeof(double));
	for (int i = 0; i <= inputSize; i++) {
		this->weights[i] = rand() % 2 - 1;
	}
}

Iris::Iris(int flowerIndex) : Input() {
	if (flowerIndex >= 0 && flowerIndex <= 149) {
		try {
			ifstream inputFile(filePath);
			string line;
			string* tokens = new string[5];
			char delimiter = ',';

			if (inputFile) {
				inputFile >> line;
			}

			for (int i = 0; i < 5; ++i) {
				tokens[i] = line.substr(0, line.find(delimiter));
				line.erase(0, line.find(delimiter) + 1);

				if( i == 4) {
					this->set_label(tokens[i]);
				}
			}

			if (tokens[4] == "Iris-setosa") {
				this->set_flag('0');
			}
			else if (tokens[4] == "Iris-virginica") {
				this->set_flag('1');
			}
			else {
				this->set_flag('2');
			}

			for (int i = 0; i < 4; ++i) {
				this->features[i] = stod(tokens[i]);
			}

			delete[] tokens;
		}
		catch (exception& e) {
			cout << e.what() << endl;
		}
	}
	else {
		throw "Error: Invalid flower index";
	}
}

double Iris::operator[](int index) {
	if (index >= 0 && index < 4) {
		return features[index];
	}
	else {
		return -1;
	}
}

NN1::NN1(int inputSize, int perceptronCount, ActivationFunction* activationFunction)
{
	this->perceptronCount = perceptronCount;
	this->perceptrons = new Perceptron[perceptronCount];

	for (int i = 0; i < perceptronCount; ++i) {
		Perceptron p(inputSize, activationFunction, i);
		this->perceptrons[i] = p;
	}
}

char NN1::evaluate(Input& input)
{
	double maxOutput = -1;
	int maxIndex = -1;
	for (int i = 0; i < this->perceptronCount; ++i) {
		double output = this->perceptrons[i].forward(input);
		if (output > maxOutput) {
			maxOutput = output;
			maxIndex = i;
		}
	}
	return static_cast<char>(maxIndex);
}

double Perceptron::forward(Input& input)
{
	double sum = this->get_weight(0);
	int i = 0;
	while (input[i] != -1) {
		sum += this->get_weight(i + 1) * input[i];
		i = i + 1;
	}
	return (*this->activationFunction)(sum);
}

double Perceptron::compute_delta(Input& input) {
	double sum = this->weights[0];
	double output = this->forward(input);
	double expected = static_cast<double>(input.get_flag());

	int i = 0;
	while (input[i] != -1) {
		sum += this->get_weight(i + 1) * input[i];
		++i;
	}

	this->delta = this->activationFunction->prim(sum) * (output - expected);
	return this->delta;
}

void Perceptron::backpropagation(Input& input, double learningRate) {
	this->compute_delta(input);
	this->weights[0] -= learningRate * this->get_delta();

	int i = 0;
	while (input[i] != -1) {
		this->weights[i + 1] -= learningRate * input[i] * this->get_delta();
		++i;
	}
}

void NN1::train(Input& input, double learningRate) {
	for (int i = 0; i < this->perceptronCount; ++i) {
		this->perceptrons[i].backpropagation(input, learningRate);
	}
}

template<class T, int size, class N>
void Training<T, size, N>::train(int K, double learningRate) {
	if (K < 1) {
		throw "Error: Parameter K must be at least 1";
	}

	for (int i = 0; i < K; ++i) {
		int randIndex = rand() % size;
		T input(randIndex);
		this->neuralNetwork->train(input, learningRate);
	}
}

template<class T, int size, class N>
int Training<T, size, N>::evaluate() {
	int correctCount = 0;

	for (int index = 0; index < size; ++index) {
		T input(index);
		char actualLabel = input.get_flag();
		char predictedLabel = this->neuralNetwork->evaluate(input) - 48;

		if (predictedLabel == actualLabel) {
			++correctCount;
		}
	}

	return correctCount;
}

static void testIrisGetInformations() {
	cout << "IRIS" << endl;
	for (int j = 0; j < irisSize; j++) {
		Iris iris(j);
		
		cout << "Label name: " << iris.get_label() << endl;
		cout << "Output value: " << iris.get_flag() << endl;
		cout << "Description: ";
		for (int u = 0; u < 4; u++) {
			cout << iris[u] << " | ";
		}
		cout << endl << "---------------------------------------" << endl;
	}
}

static void testIrisNeuralNetwork(ActivationFunction* activationFunction) {
	NN1* neuralNetwork = new NN1(4, 3, activationFunction);
	Training<Iris, irisSize, NN1> training(neuralNetwork);
	training.train(10000, 0.01);
	int evaluation = training.evaluate();
	cout << "Evaluate Iris NN1: " << evaluation << endl;
}

int main(int argc, const char* argv[]) {
	testIrisGetInformations();
	testIrisNeuralNetwork(new Sigmoid());
	testIrisNeuralNetwork(new Tanh());
	return 0;
}