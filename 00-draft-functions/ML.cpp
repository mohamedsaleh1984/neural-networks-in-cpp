
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
using namespace std;

// ============================================================
// 1. ReLU (Rectified Linear Unit)
// ============================================================
namespace ReLU {
	// f(x) = max(0, x)
	double forward(double x) {
		return std::max(0.0, x);
	}

	// f'(x) = 1 if x > 0, else 0 (at x=0 we use 0)
	double derivative(double x) {
		return x > 0.0 ? 1.0 : 0.0;
	}

	// Vectorized versions
	std::vector<double> forward(const std::vector<double>& x) {
		std::vector<double> y(x.size());
		for (int i = 0; i < x.size(); i++)
			y[i] = forward(x[i]);
		return y;
	}

	std::vector<double> derivative(const std::vector<double>& x) {
		std::vector<double> y(x.size());
		for (int i = 0; i < x.size(); i++)
			y[i] = derivative(x[i]);

		return y;
	}
}

// ============================================================
// 2. Sigmoid (Logistic)
// ============================================================
namespace Sigmoid {
	// f(x) = 1 / (1 + e^(-x))
	double forward(double x) {
		// Numerically stable version to avoid overflow
		if (x >= 0) {
			double z = std::exp(-x);
			return 1.0 / (1.0 + z);
		}
		else {
			double z = std::exp(x);
			return z / (1.0 + z);
		}
	}

	// f'(x) = f(x) * (1 - f(x))
	double derivative(double x) {
		double s = forward(x);
		return s * (1.0 - s);
	}

	std::vector<double> forward(const std::vector<double>& x) {
		std::vector<double> y(x.size());

		for (int i = 0; i < x.size(); i++)
			y[i] = forward(x[i]);
		return y;
	}

	std::vector<double> derivative(const std::vector<double>& x) {
		std::vector<double> y(x.size());
		for (int i = 0; i < x.size(); i++)
			y[i] = derivative(x[i]);
		return y;
	}
}
// ============================================================
// 3. Tanh (Hyperbolic Tangent)
// ============================================================
namespace Tanh {
	// f(x) = tanh(x)
	double forward(double x) {
		return std::tanh(x);
	}

	// f'(x) = 1 - tanh^2(x)
	double derivative(double x) {
		double t = std::tanh(x);
		return 1.0 - t * t;
	}

	std::vector<double> forward(const std::vector<double>& x) {
		std::vector<double> y(x.size());
		for (int i = 0; i < x.size(); i++)
			y[i] = forward(x[i]);
		return y;
	}

	std::vector<double> derivative(const std::vector<double>& x) {
		std::vector<double> y(x.size());
		for (int i = 0; i < x.size(); i++)
			y[i] = derivative(x[i]);
		return y;
	}
}

// ============================================================
// 4. Softmax
// ============================================================
namespace Softmax {
	// f(x_i) = e^(x_i) / sum_j e^(x_j)   (numerically stable)
	std::vector<double> forward(const std::vector<double>& x) {
		if (x.empty()) return {};

		// Subtract max for numerical stability
		double maxVal = *std::max_element(x.begin(), x.end());
		std::vector<double> expVals(x.size());
		double sum = 0.0;

		for (size_t i = 0; i < x.size(); ++i) {
			expVals[i] = std::exp(x[i] - maxVal);
			sum += expVals[i];
		}
		for (size_t i = 0; i < x.size(); ++i) {
			expVals[i] /= sum;
		}
		return expVals;
	}

	// Jacobian: d f_i / d x_j = f_i * (delta_ij - f_j)
	// Returns full Jacobian matrix (n x n)
	std::vector<std::vector<double>> derivative(const std::vector<double>& x) {
		std::vector<double> f = forward(x);
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

// ============================================================
// 5. Leaky ReLU (with alpha parameter)
// ============================================================
class LeakyReLU {
public:
	explicit LeakyReLU(double alpha = 0.01) : alpha_(alpha) {
		if (alpha <= 0.0 || alpha >= 1.0)
			throw std::invalid_argument("alpha must be in (0, 1)");
	}

	// f(x) = x if x > 0, else alpha * x
	double forward(double x) const {
		return x > 0.0 ? x : alpha_ * x;
	}

	// f'(x) = 1 if x > 0, else alpha
	double derivative(double x) const {
		return x > 0.0 ? 1.0 : alpha_;
	}

	std::vector<double> forward(const std::vector<double>& x) const {
		std::vector<double> y(x.size());
		for (size_t i = 0; i < x.size(); ++i) y[i] = forward(x[i]);
		return y;
	}

	std::vector<double> derivative(const std::vector<double>& x) const {
		std::vector<double> y(x.size());
		for (size_t i = 0; i < x.size(); ++i) y[i] = derivative(x[i]);
		return y;
	}

	double alpha() const { return alpha_; }

private:
	double alpha_;
};

void printVector(const std::string& name, const std::vector<double>& v) {
	std::cout << name << ": [ ";
	for (double val : v) std::cout << std::fixed << std::setprecision(4) << val << " ";
	std::cout << "]\n";
}

// ============================================================
// Demo / Usage
// ============================================================
int test_Activitaion() {
	std::vector<double> x = { -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0 };

	std::cout << "===== Input =====\n";
	printVector("x", x);

	std::cout << "\n===== 1. ReLU =====\n";
	printVector("f(x) ", ReLU::forward(x));
	printVector("f'(x)", ReLU::derivative(x));

	std::cout << "\n===== 2. Sigmoid =====\n";
	printVector("f(x) ", Sigmoid::forward(x));
	printVector("f'(x)", Sigmoid::derivative(x));

	std::cout << "\n===== 3. Tanh =====\n";
	printVector("f(x) ", Tanh::forward(x));
	printVector("f'(x)", Tanh::derivative(x));

	std::cout << "\n===== 4. Softmax =====\n";
	printVector("f(x) ", Softmax::forward(x));
	auto J = Softmax::derivative(x);
	std::cout << "Jacobian (7x7):\n";
	for (const auto& row : J) {
		std::cout << "  [ ";
		for (double v : row) std::cout << std::fixed << std::setprecision(4) << v << " ";
		std::cout << "]\n";
	}

	std::cout << "\n===== 5. Leaky ReLU (alpha=0.01) =====\n";
	LeakyReLU lrelu(0.01);
	printVector("f(x) ", lrelu.forward(x));
	printVector("f'(x)", lrelu.derivative(x));

	// Example: Multi-class classification output
	std::cout << "\n===== Usage Example: Multi-class Output =====\n";
	std::vector<double> logits = { 2.0, 1.0, 0.1 };  // raw scores
	auto probs = Softmax::forward(logits);
	printVector("logits     ", logits);
	printVector("probabilities", probs);

	// Example: Binary classification output
	std::cout << "\n===== Usage Example: Binary Output =====\n";
	double logit = 1.5;
	std::cout << "sigmoid(1.5) = " << Sigmoid::forward(logit) << "\n";

	return 0;
}
