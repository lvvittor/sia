import math
import numpy as np

class MultilayerPerceptron():
    def __init__(
        self, learning_rate: float, inputs: np.array, hidden: np.array, expected_outputs: np.array
    ):
        """Constructor method

        Args:
            learning_rate (float): learning rate of the perceptron
            inputs (np.array): inputs of the perceptron (x_1, x_2, ..., x_n)
            expected_outputs (np.array): expected outputs of the perceptron (y_1, y_2, ..., y_n)
        """
        self.learning_rate = learning_rate

        # add bias x_0 = 1 to each input => (1, x_1, x_2, ..., x_n)
        self.inputs = np.insert(inputs, 0, 1, axis=1)
        self.expected_outputs = expected_outputs

        # first weight is the bias => (w_0, w_1, w_2, ..., w_n)
        # for each hidden layer, create a weights matrix
        self.weights = []
        weights_aux = np.zeros((hidden[0], self.inputs.shape[1]))
        self.weights.append(weights_aux)
        for idx in range(len(hidden)):
            if idx == 0:
                continue
            weights_aux = np.zeros((hidden[idx], hidden[idx-1] + 1)) # + 1 for bias
            self.weights.append(weights_aux)

        # weight matrix for last layer
        self.output_weights = np.zeros((self.expected_outputs.shape[1], hidden[-1] + 1)) # + 1 for bias
        self.historical_weights = []
        self.historical_outputs = []
        

    def activation_func(self, value):
        return 1 / (1 + math.exp(value))


    def feedforward(self):
        self.hidden_outputs = []

        # For the first hidden layer, multiply all inputs with all first weights and for each output apply the activation function
        hidden_output = np.matmul(self.weights[0], self.inputs.T).T
        hidden_output = np.vectorize(self.activation_func)(hidden_output)
        # Add 1 for later bias calculation
        hidden_output = np.insert(hidden_output, 0, 1, axis=1)
        self.hidden_outputs.append(hidden_output)
        print(f"First hidden output is {hidden_output}")

        for idx in range(len(self.weights)):
            if idx == 0:
                continue
            # For each hidden layer between the first one and the output one multiply the previous output and the current weight matrix and apply activation func
            hidden_output = np.matmul(self.weights[idx], hidden_output.T).T
            hidden_output = np.vectorize(self.activation_func)(hidden_output)

            # Add 1 for later bias calculation
            hidden_output = np.insert(hidden_output, 0, 1, axis=1)
            self.hidden_outputs.append(hidden_output)
            print(f"{idx} hidden output is {hidden_output}")

        # For the last layer (output) multiply the last output from the hidden layers and multiply for the last matrix of weights and apply the activation function
        output = np.matmul(self.output_weights, hidden_output.T).T
        output = np.vectorize(self.activation_func)(output)
        print(f"Final output is {output}")
        return output


    def backpropagation(self):
        output = self.feedforward()
        output_errors = self.expected_outputs - output

        # Remove w0 from weights and multiply by the errors to get the error provided by each hidden node
        hidden_errors = np.matmul(self.output_weights.T, output_errors)
    
        # Calculate first delta w, using [0] to test only one input
        # output[0] * (1 - output[0]) is the gradient
        deltaw = self.learning_rate * output_errors[0] * output[0] * (1 - output[0]) * self.hidden_outputs[-1][0].T
        self.output_weights = self.output_weights + deltaw


        # Assuming there is only one hidden layer only to test XOR perceptron
        # the gradient is self.hidden_outputs[-1][0] * (1 - self.hidden_outputs[-1][0]) 
        deltaw2 = self.learning_rate * hidden_errors * self.hidden_outputs[-1][0] * (1 - self.hidden_outputs[-1][0]) * self.inputs[0].T

        self.weights[0] = self.weights[0] + deltaw2