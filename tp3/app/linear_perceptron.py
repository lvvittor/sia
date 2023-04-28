import numpy as np
from perceptron import Perceptron

class LinearPerceptron(Perceptron):

  def activation_func(self, value):
    return value # identity function
  
  
  def get_outputs(self):
    """Returns the perceptron's output for each input"""

    # Compute the perceptron's excitation for each input, including the sum of the bias
    excitations = np.dot(self.inputs, self.weights)

    # Apply the activation function to each element of the array
    return np.vectorize(self.activation_func)(excitations)
  
  
  def update_weights(self):
    # Get the difference between the expected outputs and the actual outputs
    output_errors = self.expected_outputs - self.get_outputs()

    # Compute the delta weights for each input
    deltas = self.learning_rate * output_errors.reshape(-1, 1) * self.inputs
    
    # Sum the delta weights for each input, and add them to the weights
    self.weights = self.weights + np.sum(deltas, axis=0)


  def get_error(self):
    # Mean Square Error - MSE
    p = self.inputs.shape[0]
    output_errors = self.expected_outputs - self.get_outputs()
    return np.power(output_errors, 2).sum() / p
  
  
  def is_converged(self):
    return self.get_error() < 0.05