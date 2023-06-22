# Autoencoders

Implemented Autoencoders:

- Basic.
- Denoising.
- Variational.

## Configuration

You'll find a `config.json` file in this folder, which looks something like this:

```
{
  "learning_rate": 0.001,
  "verbose": false,
  "epochs": 50000,
  "exercise": 1,
  "optimization": "adam",
  "loss_function": "bce",
  "denoising_autoencoder": {
    "execute": true,
    "train_noise": 0.2,
    "predict_rounds": 5,
    "data_augmentation_factor": 10,
    "predict_noises": [0.1, 0.3, 0.5, 0.7, 0.9]
  },
  "middle_point": {
    "execute": false,
    "first_input_index": 5,
    "second_input_index": 15
  },
  "adam_optimization": {
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8
  }
}
```

- `learning_rate`: Specifies the learning rate used for training the model. It controls the step size at each iteration during gradient descent or optimization. In this case, the learning rate is set to 0.001.

- `verbose`: Determines whether or not to display detailed progress and debugging information during the training process. If set to true, additional information will be printed. Otherwise, if set to false, the output will be more concise.

- `epochs`: Indicates the number of training epochs, which represents the number of times the model will iterate over the entire dataset. In this configuration, the model will be trained for 50,000 epochs.

- `exercise`: Represents the specific exercise or task being performed. 

- `optimization`: Specifies the optimization algorithm used for training the model. In this case, the "adam" optimizer is utilized. Adam (short for Adaptive Moment Estimation) is an optimization algorithm that combines ideas from both AdaGrad and RMSProp, providing efficient and effective optimization.

- `loss_function`: Determines the loss function utilized during training. The "bce" value stands for binary cross-entropy, a commonly used loss function for binary classification tasks.

- `denoising_autoencoder`: This section contains parameters related to a denoising autoencoder, a type of neural network used for noise reduction.

    `execute`: Specifies whether or not to execute the denoising autoencoder. If set to true, the denoising autoencoder will be trained and executed.

    `train_noise`: Represents the level of noise added to the training data during the denoising autoencoder training phase.

    `predict_rounds`: Specifies the number of rounds or iterations for which the denoising autoencoder will perform predictions on the data.

    `data_augmentation_factor`: Represents a factor by which the training data is augmented. This means that the training data is artificially expanded by a factor of 10.

    `predict_noises`: Contains a list of different noise levels for which the denoising autoencoder will make predictions. For each noise level in the list (0.1, 0.3, 0.5, 0.7, and 0.9), the denoising autoencoder will generate denoised outputs.

- `middle_point`: This section pertains to the middle point calculation within the overall process.

    `execute`: Determines whether or not to execute the middle point calculation. If set to true, the middle point calculation will be performed.

    `first_input_index`: Represents the index of the first input to be used in the middle point calculation. In this case, the index is set to 5.

    `second_input_index`: Represents the index of the second input to be used in the middle point calculation. Here, the index is set to 15.

- `adam_optimization`: This section contains parameters specific to the Adam optimization algorithm.

    `beta1`: Represents the exponential decay rate for the first moment estimates in Adam. It is set to 0.9.

    `beta2`: Represents the exponential decay rate for the second moment estimates in Adam. It is set to 0.999.

    `epsilon`: A small value used for numerical stability in Adam optimization. It prevents division by zero and is set to 1e-8.

## Running the project

Please follow the instructions on the `README.md` located in the parent folder to run the project using either `Docker` or `Poetry`.
