import numpy as np

from visualization import visualize_3d
from step_perceptron import StepPerceptron
from linear_perceptron import LinearPerceptron
from settings import settings
from utils import logical_and, logical_xor, parse_csv

def main():
  # Run excercise 1
  # exercise_1()

  # Run excercise 2
  exercise_2()


def exercise_1():
  inputs = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])

  # Test logical AND
  expected_outputs = np.apply_along_axis(logical_and, axis=1, arr=inputs)

  step_perceptron = StepPerceptron(settings.learning_rate, inputs, expected_outputs)

  epochs = step_perceptron.train(30)

  print(f"Finished learning AND at {epochs} epochs")
  print("Output: ", step_perceptron.get_outputs())
  print("Weights: ", step_perceptron.weights)

  print("\n")

  # Test logical XOR
  expected_outputs = np.apply_along_axis(logical_xor, axis=1, arr=inputs)

  step_perceptron = StepPerceptron(settings.learning_rate, inputs, expected_outputs)

  epochs = step_perceptron.train(30)

  print(f"Finished learning XOR at {epochs} epochs")
  print("Output: ", step_perceptron.get_outputs())
  print("Weights: ", step_perceptron.weights)


def exercise_2():
  inputs, expected_outputs = parse_csv(f"{settings.Config.data_path}/regression_data.csv")

  # visualize_3d(inputs)

  point_amt = 5

  print(f"{inputs[:point_amt]=}")
  print(f"{expected_outputs[:point_amt]=}\n\n")

  linear_perceptron = LinearPerceptron(settings.learning_rate, inputs[:point_amt], expected_outputs[:point_amt])

  epochs = linear_perceptron.train(10)

  print(f"Finished learning at {epochs} epochs")
  print("Output: ", linear_perceptron.get_outputs())
  print(f"Expected: {expected_outputs[:point_amt]}")
  print("Weights: ", linear_perceptron.weights)

if __name__ == "__main__":
  main()
