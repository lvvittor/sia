import numpy as np

from mlp import MLP
from settings import settings
from autoencoder import Autoencoder

class VariationalAutoencoder(Autoencoder):

    def __init__(self, inputs: np.array, hidden_nodes: list[int], latent_dim: int):
        super().__init__(inputs, hidden_nodes, latent_dim)

        # The output layer of the encoder will output the mean and variance of the latent vector
        self.encoder = MLP([inputs.shape[1], *hidden_nodes, latent_dim * 2], "linear")


    def train(self, epochs: int):
        for epoch in range(1, epochs):
            # Forward pass
            H_enc, V_enc, hO_enc, O_enc = self.encoder.feed_forward(self.inputs)

            # Stochastic layer
            latent_dim = O_enc.shape[1] // 2
            # Encoder output layer is split in two: half for mean, half for variance
            z_mean = O_enc[:, :latent_dim]     # mean of latent vector
            z_log_var = O_enc[:, latent_dim:]  # log variance of latent vector
            z = self.sampling(z_mean, z_log_var)

            H_dec, V_dec, hO_dec, O = self.decoder.feed_forward(z)

            # Backward pass
            total_loss, d_reconstruction_loss, d_kl_loss = self._vae_loss(self.inputs, O, z_mean, z_log_var)

            # TODO: see if it's correct passing only the reconstruction loss to the decoder
            delta_sum = self.decoder.backward_propagation(epoch, H_dec, V_dec, hO_dec, d_reconstruction_loss)

            # TODO: Backpropagation through stochastic layer

            # TODO: Backpropagation through encoder


    def sampling(self, z_mean: np.array, z_log_variance: np.array):
        # epsilon ~ N(0, 1)
        epsilon = np.random.normal(size=z_mean.shape)
        # Normal distribution with mean `z_mean`and variance `z_log_variance`
        return z_mean + epsilon * np.exp(z_log_variance / 2) # h(z) = mu + sigma * epsilon
    

    def _vae_loss(self, X, X_gen, z_mean, z_log_var):
        """Compute the loss between the original inputs `X` and the generated (new) outputs `X_gen` of the VAE."""

        # Total loss
        reconstruction_loss = self._binary_cross_entropy(X, X_gen)
        kl_loss = self._kl_divergence(z_mean, z_log_var)
        total_loss = np.mean(reconstruction_loss + kl_loss)

        # Gradient loss
        d_reconstruction_loss = self._binary_cross_entropy_derivative(X, X_gen)
        d_kl_loss = self._kl_divergence_derivative(z_mean, z_log_var)

        return total_loss, d_reconstruction_loss, d_kl_loss
    

    def _kl_divergence(self, z_mean: np.array, z_log_var: np.array):
        return -0.5 * np.mean(np.sum(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=-1))


    def _kl_divergence_derivative(self, z_mean: np.array, z_log_var: np.array):
        return -0.5 * (1 + z_log_var - np.square(z_mean) - np.exp(z_log_var))


    def predict(self, inputs: np.array):
        pass
