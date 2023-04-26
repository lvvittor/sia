import numpy as np

from step_perceptron import StepPerceptron
from settings import settings
from utils import logic_and

if __name__ == "__main__":
  inputs = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
  # expected_outputs = np.vectorize(logic_and)(inputs)
  expected_outputs = np.apply_along_axis(logic_and, axis=1, arr=inputs)
  step_perceptron = StepPerceptron(settings.learning_rate, inputs, expected_outputs)

  epochs = step_perceptron.train()
  print(f"Finished at {epochs} epochs")

  print("Outputs: ", step_perceptron.get_outputs())
