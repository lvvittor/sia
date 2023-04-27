import numpy as np

from simple_perceptron import SimplePerceptron
from settings import settings

if __name__ == "__main__":
    # X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    # y = np.array([-1, -1, -1, 1])
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, 1])
    perceptron = SimplePerceptron(eta=settings.learning_rate, epochs=settings.epochs)
    perceptron.fit(X, y)
