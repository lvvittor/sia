from settings import settings
import numpy as np
from pprint import pprint

class DiscreteHopfield:
    def __init__(self, XI: np.ndarray, ZETA: np.ndarray):
        """
            XI: Patterns to be stored in the network
            ZETA: Input pattern to be recognized from the network
        """
        if not self._is_stable(XI):
            # TODO: Raise exception here!
            print("The network is not stable")
        self.P = XI.shape[0]                # Number of patterns
        self.W = np.dot(XI, XI.T) / self.P  # Weights matrix
        np.fill_diagonal(self.W, 0)         # Weights matrix diagonal is 0 (no self connections)
        self.S = [ZETA]                     # List of the states of the network (initial state is ZETA, the last state is the current state). 
        self.N = self.S[-1].shape[0]        # Number of neurons
        self.t = 0                          # Time step
        print(f" W shape: {self.W.shape}")
        print(f" S inital state shape: {self.S[-1].shape}")

    @property
    def energy(self) -> float:
        """Returns the energy of the network"""
        return -0.5 * np.einsum("ij,i,j", self.W, self.S[-1], self.S[-1])

    @property
    def converged(self) -> bool:
        """Returns true if the network has converged"""
        return (len(self.S) >= 2 and np.array_equal(self.S[-1], self.S[-2])) or self.t >= settings.max_epochs

    def train(self):
        """Trains the network
        
        Returns:
            S: The final state of the network
            energy: The energy of the network
            iterations: The number of iterations it took to converge
        """
        while not self.converged:
            print(f"t: {self.t}, S shape: {self.S[-1].shape}, h: {self.activation_function}")
            self.S.append(self.activation_function)
            self.t += 1

        return self.S[-1], self.energy, self.t

    
    def _is_stable(self, vectors: np.ndarray):
        """Returns true if the vectors are orthogonal"""
        dot_products = np.dot(vectors, vectors.T)
        return np.allclose(dot_products, 0)

    @property
    def activation_function(self):
        """activation function of the last state of the network"""
        # TODO: Check if this is function can be vectorized
        h = np.zeros(self.N)
        print(f" h shape: {h.shape}")

        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    h[i] += self.W[i, j] * self.S[-1][j]
        
        return np.sign(h)
