import numpy as np

from mlp import MLP
from settings import settings

class Autoencoder():

    def __init__(self, inputs: np.array, hidden_nodes: int, latent_dim: int):
        self.inputs = inputs

        self.encoder = MLP(inputs.shape[1], hidden_nodes, latent_dim, "linear")

        self.decoder = MLP(latent_dim, hidden_nodes, inputs.shape[1], "sigmoid")
    

    def train(self, epochs: int):
        for epoch in range(epochs):
            # Forward pass
            h1, V1, h2, latent_vector = self.encoder.feed_forward(self.inputs)
            h3, V3, h4, O = self.decoder.feed_forward(latent_vector)

            # Backward pass
            error = self.inputs - O # TODO: should this use the loss function?
            delta_sum = self.decoder.backward_propagation(latent_vector, h3, V3, h4, error)
            self.encoder.backward_propagation(self.inputs, h1, V1, h2, delta_sum)
            
            if epoch % 1000 == 0 and settings.verbose:
                print(f"{epoch=} ; error={self.get_error(O)}")


    # TODO: change to binary cross-entropy loss function
    def get_error(self, O):
        """Mean Squared Error - MSE"""
        N = self.inputs.shape[0]
        errors = self.inputs - O # inputs are the expected outputs
        return np.power(errors, 2).sum() / N
