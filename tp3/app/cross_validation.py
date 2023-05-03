import math
import numpy as np
from linear_perceptron import LinearPerceptron
from non_linear_perceptron import NonLinearPerceptron
from settings import settings

# 4-Fold Cross Validation
def train_non_linear_cross_validation(inputs, expected_outputs):
    """
        Divide input array in N pieces, where 1 piece is for testing, N-1 pieces is for training
        Test with each piece and check which piece trained better.
    """
    permutation = np.random.permutation(inputs.shape[0])
    inputs = inputs[permutation]
    expected_outputs = expected_outputs[permutation]
    fold_amount = 4
    best_fold = -1
    amount_correct = -1
    best_error = -1
    best_input = []
    best_input_output = []
    best_test = []
    best_test_output = []
    for i in range(fold_amount):
        left_index = i * math.floor(inputs.shape[0] / fold_amount)
        right_index = (i + 1) * math.floor(inputs.shape[0] / fold_amount)

        test_input = inputs[left_index:right_index]
        test_expected_output = expected_outputs[left_index:right_index]
        train_input = [*inputs[:left_index], *inputs[right_index:]]
        train_expected_output = [*expected_outputs[:left_index], *expected_outputs[right_index:]]

        sigmoid_beta = 1
        sigmoid_func = lambda value: np.tanh(sigmoid_beta * value)  # tanh
        sigmoid_func_img = (-1, 1)
        sigmoid_func_derivative = lambda value: sigmoid_beta * (
            1 - sigmoid_func(value) ** 2
        )

        non_linear_perceptron = NonLinearPerceptron(
            settings.learning_rate, 
            train_input, 
            train_expected_output,       
            sigmoid_func=sigmoid_func,
            sigmoid_func_img=sigmoid_func_img,
            sigmoid_func_derivative=sigmoid_func_derivative
        )

        epochs, _, = non_linear_perceptron.train(settings.non_linear_perceptron.epochs)
        test_output = non_linear_perceptron.predict(test_input)
        errors = abs((test_expected_output - test_output)/test_expected_output)
        corrects = np.count_nonzero(errors < 0.05)
        non_linear_error = non_linear_perceptron.get_error()
        print(f"non linear error is {non_linear_error}")

        p = test_output.shape[0]
        mse_test_errors = test_expected_output - test_output
        mse_error = np.power(mse_test_errors, 2).sum() / p
        print(f"test set error is {mse_error}")

        print("test_output")
        print(test_output)
        print("test_expected_output")
        print(test_expected_output)


        if amount_correct == -1 or amount_correct < corrects or (amount_correct == corrects and non_linear_error < best_error ):
            best_fold = i
            best_error = non_linear_error
            amount_correct = corrects
            best_input = train_input
            best_input_output = train_expected_output
            best_test = test_input
            best_test_output = test_expected_output

        
        print(f"Done with fold {i} with {corrects} corrects and accuracy of {corrects/test_output.shape[0]} in epochs {epochs}")
        print(f"\n{test_input=}")
        print("\n-------------------\n")

    print(f"Best fold is {best_fold} with {amount_correct} corrects")
    return best_input, best_input_output, best_test, best_test_output