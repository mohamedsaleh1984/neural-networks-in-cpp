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
using namespace std;

using LossFunctionCallback = double(*)(vector<double>&, vector<double>&);

namespace data_layer {

	vector<vector<int>>  data_range = { {0,49},{50,99},{100,149} };

    struct iris {
        double sepalLength;
        double sepalWidth;
        double petalLength;
        double petalWidth;
        string name;

        std::string toString() {
            return "SepalLength " + std::to_string(sepalLength) +
					" SepalWidth " + std::to_string(sepalWidth) +
					" PetalLength " + to_string(petalLength) +
					" PetalWidth " + to_string(petalWidth) +
					" Name " + name;

        }
    };

    std::istream& operator>>(std::istream& is, iris& ir) {
        std::string line;

        // Read the entire line from the input stream
        if (std::getline(is, line)) {
            std::istringstream line_stream(line);

            // Parse the individual space-separated fields from the line
            if (!(line_stream >>
                ir.sepalLength >>
                ir.sepalWidth >>
                ir.petalLength >>
                ir.petalWidth >>
                ir.name))
            {
                // Set the stream's fail state if parsing fails
                is.setstate(std::ios_base::failbit);
            }
        }
        return is;
    }

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

	vector<double> irisToDblVector(iris ir) {
		return {
			ir.sepalLength,
			ir.sepalWidth,
			ir.petalLength,
			ir.sepalWidth
		};
	}
	
	/// <summary>
	/// Constructs a training dataset by selecting specific entries from an iris dataset and converting each to a numeric vector.
	/// </summary>
	/// <param name="dataset">A vector of iris objects representing the full dataset. Elements at indices 0–39, 50–89, and 100–139 are selected and converted for training.</param>
	/// <returns>A vector of numeric feature vectors (vector<vector<double>>), where each inner vector is the result of irisToDblVector for a selected iris instance.</returns>
	vector<vector<double>> get_training_dataset(const vector<iris>& dataset) {
		vector<vector<double>> training;
		for (int j = 0; j < data_range.size(); j++) {
			for (int i = 0; i < dataset.size(); i++) {
				if (i >= data_range[j][0]  && i <= (data_range[j][1]-10) ) {
					training.push_back(irisToDblVector(dataset[i]));
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
	vector<vector<double>> get_test_dataset(const vector<iris>& dataset) {
		vector<vector<double>>  test;
		for (int j = 0; j < data_range.size(); j++) {	
			for (int i = 0; i < dataset.size(); i++) {

				if (i >= (data_range[j][0] + 40)&& i <= data_range[j][1] ) {
					test.push_back(irisToDblVector(dataset[i]));
				}
			}
		}
		return test;
	}

	double nameToIndex(string name) {
		if (name == "Iris-setosa") 
			return 0;
		if (name == "Iris-versicolor") 
			return 1;
		if (name == "Iris-virginica")
			return 2;

		throw invalid_argument("name parameter si wrong");
		return -1;
	}

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
	
	vector<vector<double>> get_training_dataset_target(const vector<iris>& dataset) {
		vector<vector<double>> training;
		for (int j = 0; j < data_range.size(); j++) {
			for (int i = 0; i < dataset.size(); i++) {
				if (i >= data_range[j][0] && i <= (data_range[j][1] - 10)) {
					training.push_back({ nameToIndex(dataset[i].name) });
				}
			}
		}
		return training;
	}

	vector<vector<double>> get_test_dataset_target(const vector<iris>& dataset) {
		vector<vector<double>>  test;
		for (int j = 0; j < data_range.size(); j++) {
			for (int i = 0; i < dataset.size(); i++) {

				if (i >= (data_range[j][0] + 40) && i <= data_range[j][1]) {
					test.push_back({ nameToIndex(dataset[i].name) });
				}
			}
		}

		return test;
	}
}

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
}

namespace loss {

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

}

namespace neural_network_layer {
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
				hidden1[j] = activitaion::relu(sum);
			}

			// Second Layer = Hidden Layer 1 -> Hidden Layer 2 using Activitaion Function ReLu
			vector<double> hidden2(layerSizes[2]);
			for (int j = 0; j < layerSizes[2]; j++) {
				double sum = bias2[j];
				for (int i = 0; i < layerSizes[1]; i++) {
					sum += hidden1[i] * weights2(i, j);
				}
				hidden2[j] = activitaion::relu(sum);
			}

			// Third Layer = Hidden Layer 2 -> Output Layer using Activitaion Function Sigmoid
			vector<double> output(layerSizes[3]);
			for (int j = 0; j < layerSizes[3]; j++) {
				double sum = bias3[j];
				for (int i = 0; i < layerSizes[2]; i++) {
					sum += hidden2[i] * weights3(i, j);
				}
				output[j] = activitaion::sigmoid(sum);
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
						hidden1[j] = activitaion::relu(sum);
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
						hidden2[j] = activitaion::relu(sum);
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
						output[j] = activitaion::sigmoid(sum);
					}


					// Loss
					for (int j = 0; j < layerSizes[3]; j++) {
						double error = targets[k][j] - output[j];
						totalError += error * error;
					}

					vector<double> outputGradiants(layerSizes[3]);
					for (int j = 0; j < layerSizes[3]; j++) {
						outputGradiants[j] = (output[j] - targets[k][j]) * activitaion::sigmoid_dydx(outputPre[j]);
					}

					vector<double> hidden2Gradiants(layerSizes[2]);
					for (int i = 0; i < layerSizes[2]; i++) {
						double error = 0;
						for (int j = 0; j < layerSizes[3]; j++) {
							error += outputGradiants[j] * weights3(i, j);
						}
						hidden2Gradiants[i] = error * activitaion::relu_dydx(hidden2Pre[i]);
					}

					vector<double> hidden1Gradiants(layerSizes[1]);
					for (int i = 0; i < layerSizes[1]; i++) {
						double error = 0;
						for (int j = 0; j < layerSizes[2]; j++) {
							error += hidden2Gradiants[j] * weights2(i, j);
						}
						hidden1Gradiants[i] = error * activitaion::relu_dydx(hidden1Pre[i]);
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

}

int main()
{
	try
	{
		vector<data_layer::iris> dataset = data_layer::read_dataset();
		cout << "Total number of rows is " << dataset.size() << endl;

		vector<vector<double>> training_data = data_layer::get_training_dataset(dataset);
		vector<vector<double>> test_data = data_layer::get_test_dataset(dataset);
		
		//vector<vector<double>> training_target = data_layer::get_training_dataset_target(dataset);
		//neural_network_layer::NeuralNetwork nn(4, 8, 4, 1);

		//// Training Phase
		//auto  start = chrono::high_resolution_clock::now();
		//nn.train(training_data, training_target, 0.01, 2000);
		//auto end = chrono::high_resolution_clock::now();
		//cout << "Total Training Time " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms\n";


		//// Testing Phase
		//vector<vector<double>> test_data = data_layer::get_test_dataset(dataset);
		//vector<vector<double>> test_data_target = data_layer::get_test_dataset_target(dataset);
	
		//for (int i = 0; i < test_data.size(); i++) {
		//	auto output = nn.forward(test_data[i]);

		//	cout << "Actual Output "
		//		<< test_data_target[i][0]
		//		<< " Predicted Value "
		//		<< output[0]
		//		<< endl;
		//}

	}
	catch (const std::exception&)
	{

	}


    return 0;
}
