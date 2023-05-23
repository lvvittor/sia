import numpy as np

from sklearn.preprocessing import StandardScaler

class Kohonen():
    def __init__(self, k, inputs):
        self.k = k                  # k^2 = amount of neurones (k x k map)
        self.p = inputs.shape[0]    # amount of inputs
        self.n = inputs.shape[1]    # dimensions of each input

        scaler = StandardScaler()
        self.inputs = scaler.fit_transform(inputs) # standardize inputs (mean=0, std=1)

        # k^2 weights, each with n dimensions (same as inputs).
        # Initialized with uniform distribution U(0,1). Could also initialize with samples from the inputs.
        self.weights = np.random.rand(self.k**2, self.n)

        self.R = 1.0          # initial radius of the neighbourhood
        self.eta = 1.0        # initial learning rate


    def train(self, max_epochs=100):
        for epoch in range(1, max_epochs):
            # Get a random input
            x = self.inputs[np.random.randint(self.p)]

            # Get the index of the minimum distance neurone (winner neurone)
            distances = np.linalg.norm(self.weights - x, axis=1) # euclidean distance between `x` and each neurone's weights
            winner_neuron_index = np.argmin(distances)

            # Get the indexes of all the neighbours of the winner neurone (inside the radius `R`)
            winner_neighbours = self.get_neighbours(winner_neuron_index, self.R) # includes the winner neurone itself

            # Update the weights of the winner neurone and its neighbours
            for neuron in winner_neighbours:
                # The learning rate decreases with the epoch
                self.weights[neuron] += (self.eta/epoch) * (x - self.weights[neuron])

        return self.weights


    def get_neighbours(self, neuron_index, radius):
        """
        Returns the indexes of all the neighbours of the neuron with index `neuron_index`.
        The neuron with index `neuron_index` is also included.
        """

        neighbours = []

        # Think of the neurons' weights as a k x k matrix
        row_i, col_i = divmod(neuron_index, self.k)

        # Optimize the search by only looking at the neighbours inside a square of side length 2R+1
        min_row = max(0, row_i - int(radius))
        max_row = min(self.k - 1, row_i + int(radius))
        min_col = max(0, col_i - int(radius))
        max_col = min(self.k - 1, col_i + int(radius))

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                distance = np.linalg.norm([row - row_i, col - col_i])  # euclidean distance between (row, col) and (row_i, col_i)
                if distance <= radius:
                    index = row * self.k + col
                    neighbours.append(index)

        return neighbours
