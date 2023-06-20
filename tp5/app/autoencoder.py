import numpy as np

from mlp import MLP
from settings import settings
import seaborn as sns
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(self, inputs: np.array, hidden_nodes: int, latent_dim: int):
        self.inputs = inputs

        self.encoder = MLP(inputs.shape[1], hidden_nodes, latent_dim, "linear")

        self.decoder = MLP(latent_dim, hidden_nodes, inputs.shape[1], "sigmoid")

        # Error through epochs for plotting and early stopping
        self.losses = np.array([])  # every 1000 epochs
        self.patience = 10  # early stopping patience

        self._latent_vector = np.array([])

    @property
    def latent_vector(self):
        return self._latent_vector

    @latent_vector.setter
    def latent_vector(self, value):
        self._latent_vector = value

    def train(self, epochs: int):
        for epoch in range(1, epochs):
            # Forward pass
            h1, V1, h2, self.latent_vector = self.encoder.feed_forward(self.inputs)
            h3, V3, h4, O = self.decoder.feed_forward(self.latent_vector)

            # Backward pass
            error = self._binary_cross_entropy_derivative(
                O
            )  # error = dE/dO, where E is the loss function (e.g. binary cross entropy or MSE)

            delta_sum = self.decoder.backward_propagation(
                epoch, self.latent_vector, h3, V3, h4, error
            )
            self.encoder.backward_propagation(epoch, self.inputs, h1, V1, h2, delta_sum)

            if epoch % 1000 == 0:
                loss = self._binary_cross_entropy(O)
                self.losses = np.append(self.losses, loss)
                if settings.verbose:
                    print(f"{epoch=} ; error={loss}\n")
                if self.early_stopping():
                    break

    def early_stopping(self, threshold: float = 0.01) -> bool:
        if self.losses.shape[0] < 2:
            return False
        # Early stopping if the loss didn't decrease by at least 1%
        early_stop = self.losses[-1] > self.losses[-2] * (1 - threshold)
        if early_stop:
            self.patience -= 1
        return self.patience == 0

    def predict(self, inputs: np.array):
        _, _, _, self.latent_vector = self.encoder.feed_forward(inputs)
        _, _, _, O = self.decoder.feed_forward(self.latent_vector)
        return np.where(O < 0.5, 0, 1)  # threshold output to 0 or 1

    def _binary_cross_entropy(self, O, epsilon=1e-15):
        P = np.clip(O, epsilon, 1 - epsilon)  # avoid division by 0
        return np.mean(-self.inputs * np.log(P) - (1 - self.inputs) * np.log(1 - P))

    def _binary_cross_entropy_derivative(self, O, epsilon=1e-7):
        P = np.clip(O, epsilon, 1 - epsilon)  # avoid division by 0
        return (P - self.inputs) / (P * (1 - P))

    def _mse(self, O):
        return np.mean(np.square(self.inputs - O))

    def _mse_derivative(self, O):
        return 2 * (O - self.inputs)

    def visualize_latent_space(self):
        plt.figure()
        sns.set_palette(sns.color_palette("viridis"))
        sns.set_theme(style="white")
        sns.set(font_scale=5, rc={"figure.figsize": (20, 20)}, style="whitegrid")

        ax = sns.scatterplot(
            x=self.latent_vector[:, 0],
            y=self.latent_vector[:, 1],
            marker="o",
            s=500,
            legend=False,
        )

        plt.title("Latent space")
        ax.set_xlabel("$\mu$")
        ax.set_ylabel("$\sigma$")
        plt.grid(True)
        plt.savefig(settings.Config.output_path + "/latent_space.png")
        plt.show()