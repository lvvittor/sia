import numpy as np

class Kohonen():
    def __init__(self, k, inputs):
        self.k = k                  # k^2 = amount of neurones (k x k map)
        self.p = inputs.shape[0]    # amount of inputs
        self.n = inputs.shape[1]    # dimensions of each input

        
        # Standardize inputs (mean=0, std=1)
        # self.inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs, axis=0)

        # Calculate the mean and standard deviation of each row
        mean = np.mean(inputs, axis=1)
        std = np.std(inputs, axis=1)
        # Subtract the mean from each row and divide by the standard deviation
        self.inputs = (inputs.T - mean).T / std[:, np.newaxis]

        # Initialize weights of each neurone with uniform distribution U(0,1).
        # self.weights = np.random.rand(self.k**2, self.n)

        # Initialize weights of each neurone with random samples from the inputs.
        self.weights = np.zeros((self.k**2, self.n))
        for i in range(self.k**2):
            self.weights[i] += self.inputs[np.random.randint(self.p)]

        self.R = self.k / 2   # initial radius of the neighbourhood
        self.eta = 1.0        # initial learning rate


    def train(self, max_epochs=100):
        for epoch in range(1, max_epochs):
            # Get a random input each epoch
            # x = self.inputs[np.random.randint(self.p)]
            
            # Iterating through each input in each epoch seems to work better than the previous option, given the current parameters (mainly eta and radius decays)
            for x in self.inputs:
                # Get the index of the minimum distance neurone (winner neurone)
                distances = np.linalg.norm(self.weights - x, axis=1) # euclidean distance between `x` and each neurone's weights
                winner_neuron_index = np.argmin(distances)

                # Adjust radius with the epoch, with a minimum of 1
                radius = self.R * (1 - epoch/max_epochs)
                radius = 1 if radius < 1 else radius

                # Get the indexes of all the neighbours of the winner neurone (inside the radius `R`)
                winner_neighbours = self.get_neighbours(winner_neuron_index, radius) # includes the winner neurone itself

                # Adjust learning rate with the epoch, starting at 1 and decreasing to 0
                eta = self.eta * (1 - epoch/max_epochs)

                # Update the weights of the winner neurone and its neighbours
                for neuron in winner_neighbours:
                    # The learning rate decreases with the epoch
                    self.weights[neuron] += eta * (x - self.weights[neuron])

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
    

    def map_inputs(self, inputs):
        """
        Map each input to its closest neurone, and return the indexes of the neurones for each input.
        """
        # Reshape the inputs and weights to have dimensions (p, 1, n)
        inputs = inputs[:, np.newaxis, :]
        weights = self.weights[np.newaxis, :, :]

        # Compute the euclidean distance between each input and each neurone's weights
        distances = np.linalg.norm(inputs - weights, axis=2)

        # Return the indices of the weights that are closest to each input.
        return np.argmin(distances, axis=1)
    

    def get_heatmap(self, inputs):
        """
        Get the heatmap of the Kohonen map (i.e. how many inputs are mapped to each neurone).
        """
        winner_neurons = self.map_inputs(inputs)

        # Count the occurrences of each neuron index
        counts = np.bincount(winner_neurons)

        # Create the new array with counts
        occurrences = np.zeros(self.weights.shape[0])
        occurrences[:len(counts)] = counts # fill the array with the counts, and leave the rest as 0

        # Reshape the array to have dimensions (k, k)
        return occurrences.reshape(self.k, self.k)
