import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

class Kohonen():
    def __init__(self, k, inputs):
        self.k = k
        self.p = inputs.shape[0]
        self.n = inputs.shape[1]
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(inputs)
        standardized_data = pd.DataFrame(data=standardized_data, columns=inputs.columns.values) 
        self.inputs = standardized_data # Inputs standarized as a dataframe 
        self.weights = np.random.rand(self.k, self.k, self.n) # Array of k*k with each element n dimensions, all randoms (could use random examples from inputs).
        self.R = 1 # R is constant 1 but it should probably be a decreasing number.
        self.eta = 0.01

    def train(self, epochs=100):
        input_np = self.inputs.to_numpy()

        for epoch in range(epochs):
            # Random Xp from inputs
            xp = input_np[np.random.randint(self.p)]

            # Obtain closest w to xp
            dist = np.linalg.norm(self.weights-xp, axis=2) # dist is a matrix k*k where each element is the distance between the neurone and xp
            min_dist_index = np.unravel_index(np.argmin(dist), dist.shape) # matrix index of the minimum distance neurone
            min_dist_row, min_dist_col = min_dist_index

            # Update weights with Kohonen rule
            # Obtain all indexes that are inside R radius from the minimum neurone
            N_k = self.get_neighbours(min_dist_row, min_dist_col)

            # For each neurone inside the radius update the weights
            for coords in N_k:
                row = coords[0]
                col = coords[1]
                # Using eta/(epoch+1) as a way to decrease the learning rate.
                self.weights[row][col] += (self.eta/(epoch+1))*(xp - self.weights[row][col])

        return self.weights

    def get_neighbours(self, row, col):
        # There is probably a better way of doing this instead of using 2 fors
        neighbours = []
        # Instead of iterating through all the matrix, iterate only in the square containing the radius R that will contain all elements inside the radius R
        for i in range(np.floor(-self.R).astype(int), np.ceil(self.R+1).astype(int)):
            for j in range(np.floor(-self.R).astype(int), np.ceil(self.R+1).astype(int)):

                # Ignore elements outside of the matrix and ignore i=0, j=0 because it is the current neurone.
                if (i == 0 and j == 0) or row + i < 0 or row + i >= self.weights.shape[0] or col + j < 0 or col + j >= self.weights.shape[1]:
                    continue

                # Calculate distance using the indexes using that adjacents elements in the matrix have distance 1
                distance = np.linalg.norm(np.array([row + i, col + j]) - np.array([row, col]))

                # If it is inside the radius, it is a neighbour.
                if distance <= self.R:
                    neighbours.append((row + i, col + j))
        return neighbours
