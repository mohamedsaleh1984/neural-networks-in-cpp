
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
namespace activitaion {
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
	inline double binaryCrossEntropy(double o, double t) {
		const double eps = 1e-12;
		o = min(max(o, eps), 1.0 - eps);
		return -(t * log(o) + (1.0 - t) * log(1.0 - o));
	}

	// f(x_i) = e^(x_i) / sum_j e^(x_j)   (numerically stable)
	std::vector<double> softmax(const std::vector<double>& x) {
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

void printVector(const std::string& name, const std::vector<double>& v) {
	std::cout << name << ": [ ";
	for (double val : v) std::cout << std::fixed << std::setprecision(4) << val << " ";
	std::cout << "]\n";
}

namespace data_layer {

	/// <summary>
	/// iris class data range
	/// </summary>
	vector<vector<int>>  data_range = { {0,49},{50,99},{100,149} };

	/// <summary>
	/// iris entity
	/// </summary>
	struct iris {
		std::vector<double> features; // [sepal_length, sepal_width, petal_length, petal_width]
		std::string label;            // Species name
		iris() {
			features.resize(4);
		}
	};

	/// <summary>
	/// Overwrite standard input to read struct data
	/// </summary>
	/// <param name="is"></param>
	/// <param name="ir"></param>
	/// <returns></returns>
	std::istream& operator>>(std::istream& is, iris& ir) {
		std::string line;

		// Read the entire line from the input stream
		if (std::getline(is, line)) {
			std::istringstream line_stream(line);

			// Parse the individual space-separated fields from the line
			if (!(line_stream >>
				ir.features[0] >>
				ir.features[1] >>
				ir.features[2] >>
				ir.features[4] >>
				ir.label))
			{
				// Set the stream's fail state if parsing fails
				is.setstate(std::ios_base::failbit);
			}
		}
		return is;
	}

	/// <summary>
	/// Read iris dataset from disk
	/// </summary>
	/// <returns></returns>
	vector<iris> read_dataset() {
		vector<iris> vecs = {};

		std::ifstream file("iris.data");
		if (!file.is_open()) {
			std::cerr << "Error opening file!\n";
			return vecs;
		}

		iris temp_iris;

		// Read the file line-by-line using our overloaded operator
		while (file >> temp_iris) {
			vecs.push_back(temp_iris);
		}
		return vecs;
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
	/// <summary>
	/// Convert class to index.
	/// </summary>
	/// <param name="name">Class name</param>
	/// <returns>index</returns>
	double nameToIndex(string name) {
		if (name == "Iris-setosa")
			return 0;

		if (name == "Iris-versicolor")
			return 1;

		if (name == "Iris-virginica")
			return 2;

		throw invalid_argument("Given name {" + name + "} parameter is wrong");
		return -1;
	}

	/// <summary>
	/// Convert index back to class.
	/// </summary>
	/// <param name="index"></param>
	/// <returns></returns>
	string nameFromIndex(double index)
	{
		if (index == 0)
			return "Iris-setosa";

		if (index == 1)
			return "Iris-versicolor";

		if (index == 2)
			return "Iris-virginica";

		throw invalid_argument("Index is not correct");
		return "";
	}
}

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
