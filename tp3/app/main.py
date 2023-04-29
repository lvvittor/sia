import numpy as np

from visualization import visualize_3d, visualize_2d
from step_perceptron import StepPerceptron
from linear_perceptron import LinearPerceptron
from non_linear_perceptron import NonLinearPerceptron
from settings import settings
from utils import logical_and, logical_xor, parse_csv


def exercise_1():
    inputs = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])

    # Test logical AND
    expected_outputs = np.apply_along_axis(logical_and, axis=1, arr=inputs)

    step_perceptron = StepPerceptron(settings.learning_rate, inputs, expected_outputs)

    epochs, converged = step_perceptron.train(settings.epochs)

    if not converged:
        print("Did not converge")
    else:
      print(f"Finished learning AND at {epochs} epochs")
      print("Output: ", step_perceptron.get_outputs())
      print("Weights: ", step_perceptron.weights)

      print("\n")

    # Test logical XOR
    expected_outputs = np.apply_along_axis(logical_xor, axis=1, arr=inputs)

    step_perceptron = StepPerceptron(settings.learning_rate, inputs, expected_outputs)

    epochs, converged = step_perceptron.train(settings.epochs)


    if not converged:
        print("Did not converge")
    else:
      print(f"Finished learning XOR at {epochs} epochs")
      print("Output: ", step_perceptron.get_outputs())
      print("Weights: ", step_perceptron.weights)


def exercise_2():
    inputs, expected_outputs = parse_csv(
        f"{settings.Config.data_path}/test_data.csv", 1
    )

    # visualize_2d(inputs, expected_outputs)

    point_amt = expected_outputs.shape[0]

    # w_1 x + w_2 y + w_0 = 0

    print(f"\nInputs: {inputs[:point_amt]}\n")
    print(f"Expected Output: {expected_outputs[:point_amt]}\n\n")

    # Test linear perceptron
    # linear_perceptron = LinearPerceptron(settings.learning_rate, inputs[:point_amt], expected_outputs[:point_amt])

    # visualize_2d(inputs[:point_amt], expected_outputs[:point_amt], [0, 0])

    # epochs = linear_perceptron.train(10000)

    # visualize_2d(inputs[:point_amt], expected_outputs[:point_amt], linear_perceptron.weights)

    # print(f"Finished learning at {epochs} epochs")
    # print("Output: ", linear_perceptron.get_outputs())
    # print("Weights: ", linear_perceptron.weights)

    # Test non-linear perceptron
    sigmoid_beta = 1
    sigmoid_func = lambda value: np.tanh(sigmoid_beta * value)  # tanh
    sigmoid_func_img = (-1, 1)
    sigmoid_func_derivative = lambda value: sigmoid_beta * (
        1 - sigmoid_func(value) ** 2
    )

    non_linear_perceptron = NonLinearPerceptron(
        settings.learning_rate,
        inputs[:point_amt],
        expected_outputs[:point_amt],
        sigmoid_func=sigmoid_func,
        sigmoid_func_img=sigmoid_func_img,
        sigmoid_func_derivative=sigmoid_func_derivative,
    )

    visualize_2d(
        inputs[:point_amt], expected_outputs[:point_amt], non_linear_perceptron.weights
    )
    epochs = non_linear_perceptron.train(1000)

    print(f"\nFinished learning at {epochs} epochs")
    print("Output: ", non_linear_perceptron.get_outputs())
    print("Weights: ", non_linear_perceptron.weights)


if __name__ == "__main__":
    match settings.exercise:
        case 1:
          exercise_1()
        case 2:
          exercise_2()
        case _:
          raise ValueError("Invalid exercise number")

