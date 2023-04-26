import numpy as np

class Perceptron():

  def __init__(self, learning_rate, inputs, expected_outputs):
    self.learning_rate = learning_rate
    self.inputs = [[1, *x] for x in inputs] # add bias
    self.expected_outputs = expected_outputs
    self.weights = np.zeros(len(self.inputs[0]))

  
  def train(self, epochs=1000):
    for epoch in range(epochs):
      print(f"{epoch=}")
      self.update_weights()
      if self.get_absolute_error() == 0:
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