#include <iostream>
#include <fstream>
#include <string>
#include <exception>
#include <random>

class Input {
private:
	std::string label;
	double flag;
public:
	Input() {}
	virtual ~Input() {}

	void set_label(std::string lab) { this->label = lab; }
	std::string get_label() { return this->label; }

	// represent the input as a numeric value for training purposes
	void set_flag(double lab) { this->flag = lab; }
	double get_flag() { return this->flag; }


	virtual double operator[](int index) = 0;
};

class Iris : public Input {
private:
	double features[4];

public:
	Iris(int flowerIndex);
	Iris() {}
	~Iris() {}
	double operator[](int index);
};

class ActivationFunction
{
public:
	virtual double operator()(double x) = 0;
	virtual double prim(double x) = 0;
};

class Tanh : public ActivationFunction {
public:
	Tanh() {}
	~Tanh() {}

	// tanh(x) = (exp(x) - exp(-x))/ (exp(x) + exp(-x))
	double operator()(double x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }

	// tanh'(x) = 1 - (tanh(x))^2)
	double prim(double x) { return 1 - pow((*this)(x), 2); }
};

class Sigmoid : public ActivationFunction {
public:
	Sigmoid() {}
	~Sigmoid() {}

	// sig(x) = 1 / (1 + exp(-x))
	double operator()(double x) {
		return 1 / (1 + exp(-x));
	}

	//sig'(x) = (1 - sig(x))
	double prim(double x) {
		return (*this)(x) * (1 - (*this)(x));
	}
};

class Perceptron {
protected:
	double* weights = NULL;
	ActivationFunction* activationFunction;
	double delta;
	std::string label;

public:
	Perceptron(int inputSize, ActivationFunction* activationFunction, char label);
	Perceptron() {
		if(this->weights != NULL) {
			delete[] this->weights;
		}
	}
	~Perceptron() {}

	double get_weight(int index) { return weights[index]; }
	double get_delta() { return delta; }
	std::string get_label() { return label; }

	double forward(Input& input);
	double compute_delta(Input& input);
	void backpropagation(Input& input, double mu);
};

class NN1 {
private:
	Perceptron* perceptrons;
	int perceptronCount;

public:
	NN1() {}
	~NN1() {}

	NN1(int inputSize, int perceptronCount, ActivationFunction* activationFunction);
	char evaluate(Input& input);
	void train(Input& input, double learningRate);
};

template <class T, int size, class N>
class Training {
private:
	N* neuralNetwork;

public:
	Training();
	Training(N* neuralNetwork);

	void train(int K, double learningRate);
	int evaluate();
};


template<class T, int size, class N>
Training<T, size, N>::Training() {}

template<class T, int size, class N>
Training<T, size, N>::Training(N* neuralNetwork) {
	this->neuralNetwork = neuralNetwork;
}



class RandomGenerator {
private:
	double lowerBound;
	double upperBound;
	std::random_device rd;

public:
	RandomGenerator(double lowerBound, double upperBound) : lowerBound(lowerBound), upperBound(upperBound) {
	}
	
	double generate() {
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(lowerBound, upperBound);
		return dis(rd);
	}
};