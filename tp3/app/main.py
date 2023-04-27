import numpy as np

from simple_perceptron import SimplePerceptron
from settings import settings

if __name__ == "__main__":
    perceptron = SimplePerceptron(eta=settings.learning_rate, epochs=settings.epochs)

    print("Logical AND")
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([-1, -1, -1, 1])
    perceptron.fit(X, y)

    print("Logical OR exclusive")
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, 1])
    perceptron.fit(X, y)
