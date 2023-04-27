import numpy as np
import math

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
          print(f"{epoch=} ; weights={self.weights} ; output={self.get_outputs()} ; error={self.get_absolute_error()}")

          self.update_weights()
          
          if math.isclose(self.get_absolute_error(), 0, abs_tol=1e-5):
            break

        return epoch + 1
    

    def get_absolute_error(self):
        return np.sum(abs(self.expected_outputs - self.get_outputs()))

    def activation_func(self, value):
        raise NotImplementedError

    def get_outputs(self):
        raise NotImplementedError

    def update_weights(self):
        raise NotImplementedError
