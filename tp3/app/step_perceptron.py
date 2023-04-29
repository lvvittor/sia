import numpy as np
from perceptron import Perceptron
from settings import settings


class StepPerceptron(Perceptron):
    def activation_func(self, value):
        return 1 if value >= 0 else -1  # step function

    def update_weights(self):
        # Get the difference between the expected outputs and the actual outputs
        output_errors = self.expected_outputs - self.get_outputs()

        # Compute the delta weights for each input
        deltas = self.learning_rate * output_errors.reshape(-1, 1) * self.inputs

        # Sum the delta weights for each input, and add them to the weights
        self.weights = self.weights + np.sum(deltas, axis=0)

    def get_error(self):
        return np.sum(abs(self.expected_outputs - self.get_outputs()))

    def is_converged(self):
        return self.get_error() == settings.step_perceptron.convergence_threshold
