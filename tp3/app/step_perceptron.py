import numpy as np
from perceptron import Perceptron

class StepPerceptron(Perceptron):

  def activation_func(self, value):
    return 1 if value >= 0 else -1
  
  
  def get_outputs(self):
    """Returns the perceptron's output for each input"""
    # Compute the perceptron's excitation for each input, and substract the bias.
    # Substract 2 * bias instead of just `bias` to account for the bias present in the `weights`.
    bias = self.weights[0]
    excitations = np.dot(self.inputs, self.weights) - 2 * bias
    # Apply the activation function to each element of the array
    return np.vectorize(self.activation_func)(excitations)
    # return np.apply_along_axis(self.activation_func, axis=1, arr=excitations)
  
  
  def update_weights(self):
    outputs = self.get_outputs()

    # Compute the delta weights for each input
    deltas = self.learning_rate * (self.expected_outputs - outputs).reshape(-1, 1) * self.inputs
    
    # Sum the delta weights for each input, and add them to the weights
    self.weights = self.weights + np.sum(deltas, axis=0)
