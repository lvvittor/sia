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
            error = self._binary_cross_entropy_derivative(O) # error = dE/dO, where E is the loss function (e.g. binary cross entropy or MSE)

            delta_sum = self.decoder.backward_propagation(latent_vector, h3, V3, h4, error)
            self.encoder.backward_propagation(self.inputs, h1, V1, h2, delta_sum)
            
            if epoch % 1000 == 0 and settings.verbose:
                print(f"{epoch=} ; error={self._binary_cross_entropy(O)}\n")
                print(f"input[0]={self.inputs[0].astype(int)}")
                print(f"output[0]={np.where(O < 0.5, 0, 1)[0]}\n\n") # threshold output to 0 or 1


    def _binary_cross_entropy(self, O, epsilon=1e-15):
        P = np.clip(O, epsilon, 1 - epsilon) # avoid division by 0
        return np.mean(-self.inputs * np.log(P) - (1 - self.inputs) * np.log(1 - P))


    def _binary_cross_entropy_derivative(self, O, epsilon=1e-7):
        P = np.clip(O, epsilon, 1 - epsilon) # avoid division by 0
        return (P - self.inputs) / (P * (1 - P))


    def _mse(self, O):
        return np.mean(np.square(self.inputs - O))


    def _mse_derivative(self, O):
        return 2 * (O - self.inputs)
