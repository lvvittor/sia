import numpy as np

from settings import settings
from utils import feature_scaling

class MultilayerPerceptron():
    """Multilayer perceptron class (1 hidden layer and 1 output layer))"""

    def __init__(
        self, learning_rate: float, inputs: np.array, hidden_nodes: int, output_nodes: int, expected_outputs: np.array
    ):
        """Constructor method

        Args:
            learning_rate (float): learning rate of the perceptron
            inputs (np.array): inputs of the perceptron [(x1_1, x1_2, ..., x1_m), (x2_1, x2_2, ..., x2_m), ..., (xn_1, xn_2, ..., xn_m)]
            expected_outputs (np.array): expected outputs of the perceptron (y_1, y_2, ..., y_n)
        """
        self.learning_rate = learning_rate

        # add bias x_0 = 1 to each input => (1, x_1, x_2, ..., x_m)
        self.X = np.insert(inputs, 0, 1, axis=1)               # XOR: shape(4, 2) , DIGITS: shape(10, 1 * 7 * 5)

        print("INPUTS SHAPE: ", self.X.shape)
        print(self.X)
        print()

        # Scale the outputs to be between 0 and 1 (same as sigmoid logistic function image)
        expected_range = (np.min(expected_outputs), np.max(expected_outputs))
        self.Y = feature_scaling(expected_outputs, expected_range, (0, 1)) # XOR: shape(1, 4) , DIGITS: shape(1, 10)

        # Amount of perceptrons in the hidden and output layers
        self.hidden_nodes = hidden_nodes                      # XOR: 2, DIGITS: 10
        self.output_nodes = output_nodes                      # XOR: 1, DIGITS: 10

        # Amount of features
        self.M = self.X.shape[1]
        
        # m0 - shape(self.hidden_nodes, self.M) , m1 - shape(output_nodes, hidden_nodes)
        self.weights = [
            # Can also try with uniform distribution between -1 or 0 and 1 to initialize the weights
            np.random.randn(self.hidden_nodes, self.M),               # XOR: shape(2, 2) , DIGITS: shape(10, 1 * 7 * 5)
            # +1 for bias => simulate that the bias is an extra hidden neuron with constant output 1
            np.random.randn(self.output_nodes, self.hidden_nodes + 1) # XOR: shape(1, 2) , DIGITS: shape(10, 10)
        ]

        print("HIDDEN NODES: ", self.hidden_nodes)
        print("OUTPUT NODES: ", self.output_nodes)
        print()

        print("WEIGHTS I2H SHAPE: ", self.weights[0].shape)
        print(self.weights[0])
        print()

        print("WEIGHTS H2O SHAPE: ", self.weights[1].shape)
        print(self.weights[1])
        print()

        # Momentum
        self.previous_deltas = [np.zeros(self.weights[0].shape), np.zeros(self.weights[1].shape)]

        self.historical_error = []


    def activation_func(self, V):
        """Sigmoid activation function (logistic function)"""
        return 1 / (1 + np.exp(-V))
    

    def activation_func_derivative(self, V):
        """Derivative of the sigmoid activation function"""
        activation_function = self.activation_func(V)
        return activation_function * (1 - activation_function)


    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  
        _, _, _, output = self.feed_forward(X)
        return output


    def feed_forward(self, X):
        """
        M          => amount of features + 1 (bias)
        weights[0] => shape(hidden_nodes, M)
        weights[1] => shape(output_nodes, hidden_nodes + 1)
        X          => shape(N, M)
        """
        h1 = self.weights[0].dot(X.T) # (hidden_nodes, M) x (M, N) = (hidden_nodes, N)
        # Hidden layer output
        V1 = self.activation_func(h1) # (hidden_nodes, N)
        # Add bias to hidden layer output
        V1 = np.insert(V1, 0, 1, axis=0) # (hidden_nodes + 1, N)

        h2 = self.weights[1].dot(V1) # (output_nodes, hidden_nodes + 1) x (hidden_nodes + 1, N) = (output_nodes, N)

        # Output layer output
        O = self.activation_func(h2) # (output_nodes, N)

        return h1, V1, h2, O


    def backward_propagation(self, h1, V1, h2, O):
        """
        N      => amount of inputs
        Y      => shape(1, N)
        h1 => shape(hidden_nodes, N)
        V1 => shape(hidden_nodes + 1, N)
        h2, O  => shape(output_nodes, N)
        """
        # Update output layer weights
        output_errors = self.Y - O                                # (1, N) - (output_nodes, N)            = (output_nodes, N), substract each output from the output layer from the expected output
        dO = output_errors * self.activation_func_derivative(h2)  # (output_nodes, N) * (output_nodes, N) = (output_nodes, N), multiply element by element
        dW = self.learning_rate * dO.dot(V1.T)                    # (output_nodes, N) x (N, hidden_nodes + 1) = (output_nodes, hidden_nodes + 1)
        
        # Update hidden layer weights
        output_layer_delta_sum = dO.T.dot(self.weights[1][:, 1:])        # (N, output_nodes) x (output_nodes, hidden_nodes) = (N, hidden_nodes) . Don't use the bias term in the calculation
        dV1 = output_layer_delta_sum.T * self.activation_func_derivative(h1) # (hidden_nodes, N) * (hidden_nodes, N) = (hidden_nodes, N)
        dw = self.learning_rate * dV1.dot(self.X)                 # (hidden_nodes, N) x (N, M) =  (hidden_nodes, M)

        if settings.optimization.active and settings.optimization.method == "momentum":
            aux_dW = dW.copy()
            aux_dw = dw.copy()
            dw += settings.optimization.momentum_rate * self.previous_deltas[0]
            dW += settings.optimization.momentum_rate * self.previous_deltas[1]
            self.previous_deltas = [aux_dw, aux_dW]

        self.weights[1] += dW
        self.weights[0] += dw


    def train(self, max_epochs: int):
        """Train the perceptron

        Args:
            max_epochs (int): maximum number of epochs
        """
        for epoch in range(max_epochs):
            h1, V1, h2, O = self.feed_forward(self.X)

            self.historical_error.append((epoch, self.get_error(O)))

            if self.is_converged(O):
                break

            if epoch % 1000 == 0 and settings.verbose:
                print(f"{epoch=} ; output={O} ; error={self.get_error(O)}")

            self.backward_propagation(h1, V1, h2, O)

        return O, epoch, self.is_converged(O)
    

    def is_converged(self, O):
        expected_outputs_amplitude = 1 - 0 # amplitude of the expected output values (scaled to logistic function range)
        percentage_threshold = settings.multilayer_perceptron.convergence_threshold / 100
        return self.get_error(O) < percentage_threshold * expected_outputs_amplitude
    
    
    def get_error(self, O):
        """Mean Squared Error - MSE"""
        p = self.X.shape[0]
        output_errors = self.Y - O
        return np.power(output_errors, 2).sum() / p