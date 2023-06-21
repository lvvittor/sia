import numpy as np

from mlp import MLP
from settings import settings

class Autoencoder():

    def __init__(self, inputs: np.array, hidden_nodes: list[int], latent_dim: int):
        self.inputs = inputs

        self.encoder = MLP([inputs.shape[1], *hidden_nodes, latent_dim], "linear")

        self.decoder = MLP([latent_dim, *hidden_nodes, inputs.shape[1]], "sigmoid")

        # Error through epochs for plotting and early stopping
        self.losses = np.array([]) # every 1000 epochs
        self.patience = 10 # early stopping patience


    def train(self, epochs: int):
        for epoch in range(1, epochs):
            # Forward pass
            H_enc, V_enc, hO_enc, latent_vector = self.encoder.feed_forward(self.inputs)
            H_dec, V_dec, hO_dec, O = self.decoder.feed_forward(latent_vector)

            # Backward pass
            error = self._binary_cross_entropy_derivative(self.inputs, O) # error = dE/dO, where E is the loss function (e.g. binary cross entropy or MSE)

            delta_sum = self.decoder.backward_propagation(epoch, H_dec, V_dec, hO_dec, error)
            self.encoder.backward_propagation(epoch, H_enc, V_enc, hO_enc, delta_sum)
            
            if epoch % 1000 == 0:
                loss = self._binary_cross_entropy(self.inputs, O)
                self.losses = np.append(self.losses, loss)
                if settings.verbose: print(f"{epoch=} ; error={loss}\n")
                if self.early_stopping(): break


    def early_stopping(self, threshold: float = 0.01) -> bool:
        if self.losses.shape[0] < 2: return False
        # Early stopping if the loss didn't decrease by at least 1%
        early_stop = self.losses[-1] > self.losses[-2] * (1 - threshold)
        if early_stop: self.patience -= 1
        return self.patience == 0


    def predict(self, inputs: np.array):
        _, _, _, latent_vector = self.encoder.feed_forward(inputs)
        _, _, _, O = self.decoder.feed_forward(latent_vector)
        return np.where(O < 0.5, 0, 1) # threshold output to 0 or 1


    def _binary_cross_entropy(self, Y, O, epsilon=1e-15):
        P = np.clip(O, epsilon, 1 - epsilon) # avoid division by 0
        return np.mean(-Y * np.log(P) - (1 - Y) * np.log(1 - P))


    def _binary_cross_entropy_derivative(self, Y, O, epsilon=1e-7):
        P = np.clip(O, epsilon, 1 - epsilon) # avoid division by 0
        return (P - Y) / (P * (1 - P))


    def _mse(self, Y, O):
        return np.mean(np.square(Y - O))


    def _mse_derivative(self, Y, O):
        return 2 * (O - Y)
