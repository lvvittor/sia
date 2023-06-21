import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from mlp import MLP
from settings import settings
from autoencoder import Autoencoder

class VariationalAutoencoder(Autoencoder):

    def __init__(self, inputs: np.array, hidden_nodes: list[int], latent_dim: int):
        super().__init__(inputs, hidden_nodes, latent_dim)

        # The output layer of the encoder will output the mean and variance of the latent vector
        self.encoder = MLP([inputs.shape[1], *hidden_nodes, latent_dim * 2], "linear")

        # Save losses for plotting
        self.kl_losses = np.array([])
        self.rec_losses = np.array([])


    def train(self, epochs: int):
        for epoch in range(1, epochs):
            # Forward pass
            H_enc, V_enc, hO_enc, O_enc = self.encoder.feed_forward(self.inputs)

            # Stochastic layer
            latent_dim = O_enc.shape[1] // 2
            # Encoder output layer is split in two: half for mean, half for variance
            z_mean = O_enc[:, :latent_dim]                # mean of latent vector
            z_log_var = O_enc[:, latent_dim:]             # log variance of latent vector
            epsilon, z = self.sampling(z_mean, z_log_var) # reparametrization trick

            H_dec, V_dec, hO_dec, O = self.decoder.feed_forward(z)

            # Backward pass
            rec_loss, kl_loss, d_reconstruction_loss = self._vae_loss(self.inputs, O, z_mean, z_log_var)

            # Reference:
            # - https://github.com/pometa0507/Variational-Autoencoder-Numpy/blob/master/VAE_Numpy.ipynb
            # - https://github.com/abhayran/VAE-numpy/blob/main/vae.py#L56
            # - https://github.com/iqDF/Numpy-Variational-Autoencoder/blob/master/vae/models/encoder_model.py#L91
            delta_sum = self.decoder.backward_propagation(epoch, H_dec, V_dec, hO_dec, d_reconstruction_loss)

            # First half of the gradients are for the mean, second half for the variance
            kl_gradients = np.append(z_mean, 0.5 * (np.exp(z_log_var) - 1), axis=1) # partial derivative of `KL_divergence` w.r.t. z_mean (left) and z_log_var (right)
            rec_gradients = np.append(delta_sum, delta_sum * epsilon * np.exp(z_log_var * 0.5) * 0.5, axis=1) # delta_sum * partial derivative of `sampling` w.r.t. z_mean (left) and z_log_var (right)
            gradients = kl_gradients + rec_gradients

            self.encoder.backward_propagation(epoch, H_enc, V_enc, hO_enc, gradients)

            if epoch % 1000 == 0:
                total_loss = np.mean(rec_loss + kl_loss)
                self.losses = np.append(self.losses, total_loss)
                self.kl_losses = np.append(self.kl_losses, kl_loss)
                self.rec_losses = np.append(self.rec_losses, rec_loss)
                if settings.verbose: print(f"{epoch=} ; error={total_loss} (rec={rec_loss} ; kl={kl_loss})\n")
                if self.early_stopping(): break
    

    def early_stopping(self, threshold: float = 0.01) -> bool:
        if self.losses.shape[0] < 2: return False
        # Early stopping if none of the losses decreases by at least 1%
        early_stop_kl = self.kl_losses[-1] > self.kl_losses[-2] * (1 - threshold)
        early_stop_rec = self.rec_losses[-1] > self.rec_losses[-2] * (1 - threshold)
        if early_stop_kl and early_stop_rec: self.patience -= 1
        return self.patience == 0


    def sampling(self, z_mean: np.array, z_log_variance: np.array):
        # epsilon ~ N(0, 1)
        epsilon = np.random.normal(size=z_mean.shape)
        # Normal distribution with mean `z_mean`and variance `z_log_variance`
        return epsilon, z_mean + epsilon * np.exp(z_log_variance / 2) # h(z) = mu + sigma * epsilon
    

    def _vae_loss(self, X, X_gen, z_mean, z_log_var):
        """Compute the loss between the original inputs `X` and the generated (new) outputs `X_gen` of the VAE."""

        # Total loss
        reconstruction_loss = self._binary_cross_entropy(X, X_gen)
        kl_loss = self._kl_divergence(z_mean, z_log_var)

        # Gradient loss
        d_reconstruction_loss = self._binary_cross_entropy_derivative(X, X_gen)

        return reconstruction_loss, kl_loss, d_reconstruction_loss
    

    def _kl_divergence(self, z_mean: np.array, z_log_var: np.array):
        # KL divergence is -0.5 * (1 + log_var - mean^2 - exp(log_var)), we take the mean over the batch
        return -0.5 * np.mean(np.sum(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=-1))


    def predict_latent(self, inputs: np.array):
        _, _, _, latent_distribution = self.encoder.feed_forward(inputs)
        # Stochastic layer
        latent_dim = latent_distribution.shape[1] // 2
        # Encoder output layer is split in two: half for mean, half for variance
        z_mean = latent_distribution[:, :latent_dim]                # mean of latent vector
        z_log_var = latent_distribution[:, latent_dim:]             # log variance of latent vector
        _, z = self.sampling(z_mean, z_log_var) # reparametrization trick
        return z


    def predict(self, latent_vector: np.array):
        _, _, _, O = self.decoder.feed_forward(latent_vector)
        return np.where(O < 0.5, 0, 1) # threshold output to 0 or 1

    def visualize_latent_space(self, latent_space: np.array, labels: list[str]):
        """Visualize the latent space and save it to a file."""
        sns.set_palette(sns.color_palette("viridis"))
        sns.set_theme(style="white")
        sns.set(font_scale=5, rc={"figure.figsize": (20, 20)}, style="whitegrid")
        predict_latent_space = np.array([[-0.25, -1.5], [-1.0, 0.5], [1, -0.5], [0.25, 1.1]])
        ax = sns.scatterplot(
            x=latent_space[:, 0],
            y=latent_space[:, 1],
            marker="o",
            s=500,
            legend=False,
            color="red"
        )
    
        for i, (x, y) in enumerate(zip(latent_space[:, 0], latent_space[:, 1])):
            plt.text(x, y+0.07, labels[i], fontsize=24, ha='right', va='top')

        ax = sns.scatterplot(
            x=predict_latent_space[:, 0],
            y=predict_latent_space[:, 1],
            marker="o",
            s=500,
            legend=False,
            color=['blue', 'green', 'orange', 'purple']
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(settings.Config.output_path + "/latent_space.png")
        plt.close()


    def visualize_all_digits(self):
        n = 20  # figure with 20x20 digits
        digit_x = 7
        digit_y = 5
        spacing = 2  # Space between images
        
        figure = np.zeros((digit_x * n + spacing * (n-1), digit_y * n + spacing * (n-1)))

        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.predict(z_sample)
                digit = x_decoded[0].reshape(digit_x, digit_y)

                # Calculate start and end positions for each image in the added space
                start_x = i * (digit_x + spacing)
                end_x = start_x + digit_x
                start_y = j * (digit_y + spacing)
                end_y = start_y + digit_y

                figure[start_x:end_x, start_y:end_y] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.grid(False)
        plt.savefig(settings.Config.output_path + "/all_digits.png")
        plt.close()
