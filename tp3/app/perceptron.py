from typing import Optional
import numpy as np

from settings import settings


class Perceptron:
    def __init__(
        self, learning_rate: float, inputs: np.array, expected_outputs: np.array
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
        self.weights = np.zeros(self.inputs.shape[1])
        self.historical_weights = []
        self.historical_outputs = []


    def train(self, epochs: Optional[int] = 1000):
        """
        Trains the perceptron for a given number of epochs

        Args:
            epochs (Optional[int]): number of epochs to train the perceptron. Defaults to 1000.

        Returns:
            int: number of epochs needed to converge
            bool: whether the perceptron converged or not
        """
        for epoch in range(epochs):
            if settings.verbose: print(f"{epoch=} ; weights={self.weights} ; output={self.get_outputs()} ; error={self.get_error()}")

            # save the weights
            self.update_weights()
            self.historical_weights.append(self.weights)
            self.historical_outputs.append(self.get_outputs())

            if self.is_converged():
                break

        return epoch + 1, self.is_converged()


    def get_outputs(self):
        """Returns the perceptron's output for each input"""

        # Compute the perceptron's excitation for each input, including the sum of the bias
        excitations = np.dot(self.inputs, self.weights)

        # Apply the activation function to each element of the array
        return np.vectorize(self.activation_func)(excitations)


    def __str__(self) -> str:
        output = "Expected - Actual\n"

        for expected, actual in zip(self.expected_outputs, self.get_outputs()):
            output += f"{expected:<10} {actual}\n"

        output += f"\nWeights: {self.weights}"

        return output


    def get_error(self):
        raise NotImplementedError

    def is_converged(self):
        raise NotImplementedError

    def activation_func(self, value):
        raise NotImplementedError

    def update_weights(self):
        raise NotImplementedError

    def save_animation(self):
        raise NotImplementedError

    def save_animation_frames(self):
        raise NotImplementedError
