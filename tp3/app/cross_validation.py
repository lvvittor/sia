import math
from linear_perceptron import LinearPerceptron
from non_linear_perceptron import NonLinearPerceptron
from settings import settings

# 4-Fold Cross Validation
def train_cross_validation(inputs, expected_outputs):
    """
        Divide input array in N pieces, where 1 piece is for testing, N-1 pieces is for training
        Test with each piece and check which piece trained better.
    """
    fold_amount = 4
    best_fold = -1
    amount_correct = -1
    for i in range(fold_amount):
        left_index = i * math.floor(inputs.shape[0] / fold_amount)
        right_index = (i + 1) * math.floor(inputs.shape[0] / fold_amount)
        test_input = inputs[left_index:right_index]
        test_expected_output = expected_outputs[left_index:right_index]
        train_input = [*inputs[:left_index], *inputs[right_index:]]
        train_expected_output = [*expected_outputs[:left_index], *expected_outputs[right_index:]]

        linear_perceptron = LinearPerceptron(settings.learning_rate, train_input, train_expected_output)
        _, _ = linear_perceptron.train(settings.linear_perceptron.epochs)

        weights = linear_perceptron.weights
        corrects = 0
        for index in range(test_input.shape[0]):
            test_output = weights[0]
            counter = 1
            for element in test_input[index]:
                test_output += weights[counter] * element
                counter += 1
            
            if math.isclose(abs(test_expected_output[index] / test_output), 1, abs_tol=1e-1) : 
                corrects += 1

        if amount_correct == -1 or amount_correct < corrects:
            best_fold = i
            amount_correct = corrects
        
        print(f"Done with fold {i}")
        print(linear_perceptron)
        print(f"\n{test_input=}")
        print("\n-------------------\n")

    print(f"Best fold is {best_fold} with {amount_correct} corrects")