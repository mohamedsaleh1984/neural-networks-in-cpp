Key Design Notes
Feature						Implementation Detail
Numerical stability			Sigmoid uses branch to avoid exp(large); Softmax subtracts max before exp
Namespaces					Stateless functions grouped in namespaces (ReLU, Sigmoid, Tanh, Softmax)
Class for LeakyReLU			Since it has a parameter α, encapsulation in a class makes sense
Vectorized overloads		Convenient std::vector<double> versions for layer-level use
Softmax derivative			Returns full Jacobian matrix (n×n), as softmax is vector→vector
Edge case at x=0			ReLU/LeakyReLU sub-gradient set to 0 / α