import numpy as np

from step_perceptron import StepPerceptron
from settings import settings
from utils import logical_and, logical_xor

if __name__ == "__main__":
  inputs = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])

  # Test logical AND
  expected_outputs = np.apply_along_axis(logical_and, axis=1, arr=inputs)

  step_perceptron = StepPerceptron(settings.learning_rate, inputs, expected_outputs)

  epochs = step_perceptron.train(10)

  print(f"Finished learning AND at {epochs} epochs")
  print("Output: ", step_perceptron.get_outputs())
  print("Weights: ", step_perceptron.weights)

  print("\n")

  # Test logical XOR
  expected_outputs = np.apply_along_axis(logical_xor, axis=1, arr=inputs)

  step_perceptron = StepPerceptron(settings.learning_rate, inputs, expected_outputs)

  epochs = step_perceptron.train(10)

  print(f"Finished learning XOR at {epochs} epochs")
  print("Output: ", step_perceptron.get_outputs())
  print("Weights: ", step_perceptron.weights)
