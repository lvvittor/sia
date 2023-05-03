import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=0.9):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.momentum = 0
        self.rmsprop = 0
        self.t = 0

    def optimize(self, parameters, gradients):
        self.t += 1

        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * gradients

        self.rmsprop = self.beta2 * self.rmsprop + (1 - self.beta2) * (gradients ** 2)

        momentum_corrected = self.momentum / (1 - self.beta1 ** self.t)
        rmsprop_corrected = self.rmsprop / (1 - self.beta2 ** self.t)

        step = self.learning_rate * momentum_corrected / (np.sqrt(rmsprop_corrected) + self.epsilon)

        parameters -= step

        self.learning_rate *= self.decay_rate
        
        return parameters