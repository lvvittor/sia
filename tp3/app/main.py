import numpy as np

from step_perceptron import StepPerceptron
from linear_perceptron import LinearPerceptron
from non_linear_perceptron import NonLinearPerceptron
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
        step_perceptron.visualize()
        print(f"Finished learning at {epochs} epochs\n")

    print(step_perceptron)

    # Test logical XOR
    expected_outputs = np.apply_along_axis(logical_xor, axis=1, arr=inputs)

    step_perceptron = StepPerceptron(settings.learning_rate, inputs, expected_outputs)

    epochs, converged = step_perceptron.train(settings.step_perceptron.epochs)

    print("\n----- XOR -----\n")

    if not converged:
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


if __name__ == "__main__":
	match settings.exercise:
		case 1:
			exercise_1()
		case 2:
			exercise_2()
		case 3:
			# Example usage:
			X, y = parse_csv(
				f"{settings.Config.data_path}/test_data.csv", 1
			)
			# X, y = np.arange(10).reshape((5, 2)), list(range(5))
			print(X)
			print(y)

			# Split the data into training and testing subsets
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

			# # Print the sizes of the training and testing subsets
			print(f"X_train: {X_train}")
			print(f"y_train: {y_train}")
			print(f"X_test: {X_test}")
			print(f"y_test: {y_test}")

		case _:
			raise ValueError("Invalid exercise number")

