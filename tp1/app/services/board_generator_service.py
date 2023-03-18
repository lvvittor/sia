import numpy as np
import pandas as pd

class BoardGeneratorService():
    def __init__(self, n, m):
        self.n = n
        self.m = m

    def generate(self):
        # Generate a board with n rows and n columns
        # Each cell has a random value between 0 and m-1
        return pd.DataFrame(np.random.randint(0, self.m, size=(self.n, self.n)))