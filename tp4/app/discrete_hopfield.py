from settings import settings
import numpy as np
from pprint import pprint

class DiscreteHopfield:
    def __init__(self, XI: np.ndarray, ZETA: np.ndarray):
        """
            XI: Patterns to be stored in the network
            ZETA: Input pattern to be recognized from the network
        """
        pprint(XI)
        self.W = np.dot(XI, XI.T) / XI.shape[0] # Weights matrix (TODO: This is not correct as this should be a 2d matrix)
        pprint(self.W)
        np.fill_diagonal(self.W, 0)             # Weights matrix diagonal is 0 (no self connections)
        self.S = [ZETA]                         # States of the network (initial state is ZETA)

    @property
    def energy(self) -> float:
        """Returns the energy of the network"""
        return -0.5 * np.einsum("ij,i,j", self.W, S, S)

    @property
    def converged(self) -> bool:
        """Returns true if the network has converged"""
        return len(self.S) >= 2 and np.array_equal(self.S[-1], self.S[-2])

    def train(self):
        """Trains the network
        
        Returns:
            S: The final state of the network
            energy: The energy of the network
            iterations: The number of iterations it took to converge
        """
        while not self.converged:
            self.S.append(np.sign(np.dot(self.W, self.S[-1])))

        return self.S[-1], self.energy, len(self.S)
