from settings import settings
import numpy as np

class Hopfield:
    def __init__(self, N: int = 100):
        """
            N: number of neurons
            W: weight matrix
            xi: vector with saved patterns
            S: vector with current state of the network
            p: number of saved patterns
            t: time step
        """
        
        self.W = settings.hopfield.weights
        self.xi = settings.hopfield.xi
        self.N = self.W.shape[0]
        self.S = -1 * np.ones(self.N)
        self.p = self.xi.shape[0]
        self.t = 0

    @property
    def energy(self):
        """Returns the energy of the network"""
        return -0.5 * np.einsum("ij,i,j", self.w, S, S)

    def build_w(self):
        num_patterns = len(self.xi)
        pattern_length = len(self.patterns[0])

        for pattern in self.patterns:
            pattern = np.array(pattern)
            # we calculate matrix W that is equal to (1/n)*pattern*pattern'
            self.weights += np.dot(pattern, pattern.T)

        self.weights /= pattern_length
        np.fill_diagonal(self.weights, 0)

    def update_states():
        idx = 0
        for state in self.S:
            # We need to compare Si agains S_{i-1}, to check is conversion was reached
            state[idx] = np.sign(np.dot(self.W, state))


