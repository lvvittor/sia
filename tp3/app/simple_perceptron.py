import numpy as np
import math


class SimplePerceptron:
    def __init__(self, eta: float = 0.1, epochs: int = 1000):
        """Constructor method

        Args:
            eta (float): learning rate of the perceptron
            epochs (int): maximum number of epochs to train the perceptron
        """
        self.eta = eta
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the perceptron

        Args:
            X (np.ndarray): The input data of shape (n_samples, m_features)
            y (np.ndarray): The expected output array of shape (n_samples,)

        Returns:
            int, bool: The number of epochs needed to train the perceptron, and whether the perceptron converged or not
        """
        X = np.insert(
            X, 0, 1, axis=1
        )  # Add bias to the input data (Without these spare bias weights, our model has quite limited “movement” while searching through solution space.)
        weights = np.zeros(X.shape[1])

        for epoch in range(self.epochs):
            predicted = []
            for i, sample in enumerate(X):
                y_hat = self.predict(sample, weights)
                predicted.append(y_hat)

                for j, _ in enumerate(weights):
                    delta = self.eta * (y[i] - y_hat) * sample[j - 1]
                    weights[j - 1] += delta

            printable_metadata = {
                "epoch": epoch,
                "weights": weights,
                "y": y,
                "y_hat": predicted,
                "accuracy": self._accuracy(y, predicted),
            }
            print(f"{printable_metadata}")

    def predict(self, X: np.ndarray, w: np.ndarray):
        """Predict the output of the perceptron for a given input

        Args:
            X (np.ndarray): The data sample of shape (m_features,)
            w (np.ndarray): The weights array of shape (m_features,)

        Returns:
            int: Return 1 if the perceptron predicts a positive output, -1 otherwise
        """

        return np.where(self._sum(X, w) >= 0, 1, -1)

    def _sum(self, X: np.array, w: np.array):
        """Compute the sum of the product of the inputs and the weights

        Args:
            X (np.array): The data sample of shape (m_features,)
            w (np.array): The weights array of shape (m_features,)

        Returns:
            int: The sum of the product of the inputs and the weights
        """
        return np.sum(np.dot(X, np.transpose(w)))

    def _accuracy(self, y: np.ndarray, y_hat: np.ndarray):
        """Calculate the accuracy of predictions for a given epoch.

        Args:
            y (np.ndarray): actual array of shape (n_samples, )
            y_hat (np.ndarray): predicted array of shape (n_samples,)

        Returns:
            float: The accuracy of the perceptron
        """
        return np.sum(y == y_hat) / y.shape[0]
