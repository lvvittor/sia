import numpy as np

from perceptron import Perceptron
from utils import feature_scaling
from settings import settings

class NonLinearPerceptron(Perceptron):

	def __init__(self, learning_rate, inputs, expected_outputs, sigmoid_func, sigmoid_func_img, sigmoid_func_derivative):
		super().__init__(learning_rate, inputs, expected_outputs)

		self.expected_range = (np.min(self.expected_outputs), np.max(self.expected_outputs))
		self.sigmoid_func = sigmoid_func
		self.sigmoid_func_img = sigmoid_func_img
		self.sigmoid_func_derivative = sigmoid_func_derivative
		self.scaled_expected_outputs = feature_scaling(self.expected_outputs, self.expected_range, self.sigmoid_func_img)


	def activation_func(self, value):
		return self.sigmoid_func(value)


	def get_scaled_outputs(self):
		outputs = self.get_outputs()
		scaled_outputs = [feature_scaling(o, self.sigmoid_func_img, self.expected_range) for o in outputs]

		return np.array(scaled_outputs)


	def get_error(self):
		"""Mean Squared Error - MSE"""
		p = self.inputs.shape[0]
		output_errors = self.scaled_expected_outputs - self.get_outputs()
		return np.power(output_errors, 2).sum() / p


	def compute_deltas(self) -> np.array:
    # Get the difference between the expected outputs and the actual outputs
		output_errors = self.scaled_expected_outputs - self.get_outputs()

		# Compute the delta weights for each input
		excitations = np.dot(self.inputs, self.weights)
		derivatives = np.vectorize(self.sigmoid_func_derivative)(excitations)

		deltas = self.learning_rate * (output_errors * derivatives).reshape(-1, 1) * self.inputs

		return deltas


	def is_converged(self):
		expected_outputs_amplitude = np.max(self.scaled_expected_outputs) - np.min(self.scaled_expected_outputs)
		percentage_threshold = settings.non_linear_perceptron.convergence_threshold / 100
		return self.get_error() < percentage_threshold * expected_outputs_amplitude
	

	def __str__(self) -> str:
		output = "Expected - Actual\n"

		for expected, actual in zip(self.expected_outputs, self.get_scaled_outputs()):
			output += f"{expected:<10} {actual}\n"

		output += f"\nWeights: {self.weights}"
		
		return output
