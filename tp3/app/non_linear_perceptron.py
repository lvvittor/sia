import numpy as np

from perceptron import Perceptron
from utils import feature_scaling

class NonLinearPerceptron(Perceptron):
  def __init__(self, inputs, expected_outputs, learning_rate, sigmoid_func, sigmoid_func_img, sigmoid_func_derivative):
    super().__init__(inputs, expected_outputs, learning_rate)
    self.expected_min = np.min(expected_outputs)
    self.expected_max = np.max(expected_outputs)
    self.sigmoid_func = sigmoid_func
    self.sigmoid_func_img = sigmoid_func_img
    self.sigmoid_func_derivative = sigmoid_func_derivative


  def activation_func(self, value):
    result = self.sigmoid_func(value)
    scaled_result = feature_scaling(result, self.sigmoid_func_img, (self.expected_min, self.expected_max))
    return scaled_result
  
  
  def update_weights(self):
    # Get the difference between the expected outputs and the actual outputs
    output_errors = self.expected_outputs - self.get_outputs()

    # Compute the delta weights for each input
    excitations = np.dot(self.inputs, self.weights)
    derivatives = np.vectorize(self.sigmoid_func_derivative)(excitations)
    deltas = self.learning_rate * (output_errors * derivatives).reshape(-1, 1) * self.inputs
    
    # Sum the delta weights for each input, and add them to the weights
    self.weights = self.weights + np.sum(deltas, axis=0)


  def get_error(self):
    # Mean Square Error - MSE
    p = self.inputs.shape[0]
    output_errors = self.expected_outputs - self.get_outputs()
    return np.power(output_errors, 2).sum() / p
  
  
  def is_converged(self):
    return self.get_error() < 0.05