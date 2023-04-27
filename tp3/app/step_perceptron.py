import numpy as np
from perceptron import Perceptron

class StepPerceptron(Perceptron):

  def activation_func(self, value):
    return 1 if value >= 0 else -1 # step function
  
  
  def get_outputs(self):
    """Returns the perceptron's output for each input"""
    bias = self.weights[0]
    # Compute the perceptron's excitation for each input, and substract the bias.
    # Substract 2 * bias instead of just `bias` to account for the bias present in the `weights`.
    excitations = np.dot(self.inputs, self.weights) - 2 * bias

    # Apply the activation function to each element of the array
    return np.vectorize(self.activation_func)(excitations)
  
  
  def update_weights(self):
    # Get the difference between the expected outputs and the actual outputs
    output_errors = self.expected_outputs - self.get_outputs()

    # Compute the delta weights for each input
    deltas = self.learning_rate * output_errors.reshape(-1, 1) * self.inputs
    
    # Sum the delta weights for each input, and add them to the weights
    self.weights = self.weights + np.sum(deltas, axis=0)
