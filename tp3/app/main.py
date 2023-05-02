import numpy as np

from step_perceptron import StepPerceptron
from linear_perceptron import LinearPerceptron
from non_linear_perceptron import NonLinearPerceptron
from multilayer_perceptron import MultilayerPerceptron
from settings import settings
from utils import logical_and, logical_xor, parse_csv, train_test_split


def exercise_1():
    inputs = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])

    # Test logical AND
    expected_outputs = np.apply_along_axis(logical_and, axis=1, arr=inputs)

    step_perceptron = StepPerceptron(settings.learning_rate, inputs, expected_outputs)

    epochs, converged = step_perceptron.train(settings.step_perceptron.epochs)
      
    print("\n----- AND -----\n")

    if not converged:
        print(f"Did not converge after {epochs} epochs\n")
    else:
      step_perceptron.save_animation()
      step_perceptron.save_animation_frames()
      print(f"Finished learning AND at {epochs} epochs")
      print("Output: ", step_perceptron.get_outputs())
      print("Weights: ", step_perceptron.weights)

    print(step_perceptron)

    # Test logical XOR
    expected_outputs = np.apply_along_axis(logical_xor, axis=1, arr=inputs)

    step_perceptron = StepPerceptron(settings.learning_rate, inputs, expected_outputs)

    epochs, converged = step_perceptron.train(settings.step_perceptron.epochs)

    print("\n----- XOR -----\n")

    if not converged:
        # step_perceptron.save_animation()
        # step_perceptron.save_animation_frames()
        print(f"Did not converge after {epochs} epochs\n")
    else:
        print(f"Finished learning at {epochs} epochs\n")

    print(step_perceptron)


def exercise_2():
	inputs, expected_outputs = parse_csv(f"{settings.Config.data_path}/regression_data.csv")

	print(f"\nInputs: {inputs}\n")

	# Test linear perceptron
	linear_perceptron = LinearPerceptron(settings.learning_rate, inputs, expected_outputs)

	epochs, converged = linear_perceptron.train(settings.linear_perceptron.epochs)

	print("\n----- LINEAR PERCEPTRON -----\n")

	if not converged:
		print(f"Did not converge after {epochs} epochs\n")
	else:
		print(f"Finished learning at {epochs} epochs\n")

	print(linear_perceptron)

	# Test non-linear perceptron
	sigmoid_beta = 1
	sigmoid_func = lambda value: np.tanh(sigmoid_beta * value)  # tanh
	sigmoid_func_img = (-1, 1)
	sigmoid_func_derivative = lambda value: sigmoid_beta * (
		1 - sigmoid_func(value) ** 2
	)

	non_linear_perceptron = NonLinearPerceptron(
		settings.learning_rate,
		inputs,
		expected_outputs,
		sigmoid_func=sigmoid_func,
		sigmoid_func_img=sigmoid_func_img,
		sigmoid_func_derivative=sigmoid_func_derivative,
	)

	epochs, converged = non_linear_perceptron.train(settings.non_linear_perceptron.epochs)

	print("\n----- NON-LINEAR PERCEPTRON -----\n")

	if not converged:
		print(f"Did not converge after {epochs} epochs\n")
	else:
		print(f"Finished learning at {epochs} epochs\n")

	# Print weights and outputs
	print(non_linear_perceptron)


def exercise_3():
	# Test XOR
	inputs = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
	expected_outputs = np.array([1, 1, -1, -1])

	multilayer_perceptron = MultilayerPerceptron(settings.learning_rate, inputs, 2, 1, expected_outputs)

	print("\n----- XOR -----\n")
	
	multilayer_perceptron.train(100000)


if __name__ == "__main__":
	match settings.exercise:
		case 1:
			exercise_1()
		case 2:
			exercise_2()
		case 3:
			exercise_3()

		case _:
			raise ValueError("Invalid exercise number")

