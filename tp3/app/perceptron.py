import numpy as np

class Perceptron:
    def __init__(self, learning_rate: float, inputs: np.array, expected_outputs: float):
        """Constructor method

        Args:
            learning_rate (float): learning rate of the perceptron
            inputs (np.array): inputs of the perceptron (x_1, x_2, ..., x_n)
            expected_outputs (_type_): expected outputs of the perceptron (y_1, y_2, ..., y_n)
        """
        self.learning_rate = learning_rate
        # add bias x_0 = 1 to each input => (1, x_1, x_2, ..., x_n)
        self.inputs = np.insert(inputs, 0, 1, axis=1)
        self.expected_outputs = expected_outputs
        # first weight is the bias => (w_0, w_1, w_2, ..., w_n)
        self.weights = np.zeros(self.inputs.shape[1])


    def train(self, epochs: int = 1000):
        for epoch in range(epochs):
          print(f"{epoch=} ; weights={self.weights} ; output={self.get_outputs()} ; error={self.get_error()}")

          self.update_weights()
          
          if self.is_converged():
            break

        return epoch + 1


    def get_outputs(self):
        """Returns the perceptron's output for each input"""

        # Compute the perceptron's excitation for each input, including the sum of the bias
        excitations = np.dot(self.inputs, self.weights)

        # Apply the activation function to each element of the array
        return np.vectorize(self.activation_func)(excitations)
    

    def get_error(self):
        raise NotImplementedError
    
    def is_converged(self):
        raise NotImplementedError

    def activation_func(self, value):
        raise NotImplementedError

    def update_weights(self):
        raise NotImplementedError
