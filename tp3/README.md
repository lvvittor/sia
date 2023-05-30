# Simple and Multilayer Perceptrons

## Configuration

You'll find a `config.json` file in this folder, which looks something like this:

```
{
  "learning_rate": 0.01,
  "verbose": true,
  "exercise": 1,
  "step_perceptron": {
    "convergence_threshold": 0,
    "epochs": 30
  },
  "linear_perceptron": {
    "convergence_threshold": 20,
    "epochs": 10000
  },
  "non_linear_perceptron": {
    "convergence_threshold": 0.0011,
    "epochs": 4000
  },
  "multilayer_perceptron": {
    "convergence_threshold": 0.4,
    "epochs": 10000,
    "predicting_digit": 2
  },
  "optimization": {
    "active": false,
    "method": "momentum",
    "momentum_rate": 0.9
  }
}

```


- `learning_rate`: A floating point value that represents the learning rate of the perceptron algorithm.
- `verbose`: A boolean value that determines whether the program should output information during training or not.
- `exercise`: An integer value that determines which exercise will be run.
- `step_perceptron`: A dictionary that contains configuration variables specific to the step perceptron algorithm.
- `convergence_threshold`: A floating point value that represents the threshold for the perceptron algorithm to consider the training data as converged.
- `epochs`: An integer value that determines the maximum number of epochs that the perceptron algorithm should execute during training.
- `linear_perceptron`: A dictionary that contains configuration variables specific to the linear perceptron algorithm.
- `convergence_threshold`: A floating point value that represents the threshold for the perceptron algorithm to consider the training data as converged.
- `epochs`: An integer value that determines the maximum number of epochs that the perceptron algorithm should execute during training.
- `non_linear_perceptron`: A dictionary that contains configuration variables specific to the non-linear perceptron algorithm.
- `convergence_threshold`: A floating point value that represents the threshold for the perceptron algorithm to consider the training data as converged.
- `epochs`: An integer value that determines the maximum number of epochs that the perceptron algorithm should execute during training.
- `multilayer_perceptron`: A dictionary that contains configuration variables specific to the multilayer perceptron algorithm.
- `convergence_threshold`: A floating point value that represents the threshold for the perceptron algorithm to consider the training data as converged.
- `epochs`: An integer value that determines the maximum number of epochs that the perceptron algorithm should execute during training.
- `predicting_digit`: An integer value that determines which digit the multilayer perceptron should predict during training and testing.
- `optimization`: A dictionary that contains configuration variables specific to the optimization technique used during training.
- `active`: A boolean value that determines whether the optimization technique should be used during training or not.
- `method`: A string value that represents the optimization technique to be used during training (e.g., "momentum", "adagrad", "rmsprop", etc.).
- `momentum_rate`: A floating point value that represents the momentum rate used in the momentum optimization technique.

## Running the project

Please follow the instructions on the `README.md` located in the parent folder to run the project using either `Docker` or `Poetry`.
