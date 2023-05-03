import numpy as np
from perceptron import Perceptron
from settings import settings


class LinearPerceptron(Perceptron):

    def activation_func(self, value):
        """Identity function"""
        return value


    def compute_deltas(self) -> np.array:
        # Get the difference between the expected outputs and the actual outputs
        output_errors = self.expected_outputs - self.get_outputs(self.inputs)
        # Compute the delta weights for each input
        deltas = self.learning_rate * output_errors.reshape(-1, 1) * self.inputs

        return deltas


    def get_error(self):
        """Mean Squared Error - MSE"""
        p = self.inputs.shape[0]
        output_errors = self.expected_outputs - self.get_outputs(self.inputs)
        return np.power(output_errors, 2).sum() / p
    

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return self.get_outputs(X) 


    def is_converged(self):
        expected_outputs_amplitude = np.max(self.expected_outputs) - np.min(self.expected_outputs)
        percentage_threshold = settings.linear_perceptron.convergence_threshold / 100
        return self.get_error() < percentage_threshold * expected_outputs_amplitude
