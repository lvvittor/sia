import numpy as np

from sklearn.preprocessing import StandardScaler
from typing import Optional
from settings import settings

class Oja():
    def __init__(self, inputs: np.array):
        self.learning_rate = 10 ** -3
        scaler = StandardScaler()
        self.inputs = scaler.fit_transform(inputs) # standardize inputs (mean=0, std=1)
        self.weights = np.random.rand(self.inputs.shape[1])


    def train(self, epochs: Optional[int] = 1000):
        for epoch in range(epochs):
            if settings.verbose: print(f"{epoch=} ; weights={self.weights} ; output={self.get_outputs()} ; error={self.get_error()}")

            self.update_weights()

        return epoch + 1
    

    def update_weights(self):
        # Oja Rule
        deltas = np.sum(self.learning_rate * self.get_outputs().reshape(-1, 1) * self.inputs, axis=0)
        denominator = np.sum((self.weights + deltas)** 2) ** 0.5

        self.weights = (self.weights + deltas)/denominator


    def get_outputs(self):
        excitations = np.dot(self.inputs, self.weights)

        return np.vectorize(self.activation_func)(excitations)
    
    
    def activation_func(self, value):
        """Identity function"""
        return value
    
