import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mlp import MLP
from settings import settings

class Autoencoder():
    def __init__(self, inputs: np.array, hidden_nodes: list[int], latent_dim: int, expected_output: np.array = None):
        self.inputs = inputs

        self.encoder = MLP([inputs.shape[1], *hidden_nodes, latent_dim], "linear")

        self.decoder = MLP([latent_dim, *hidden_nodes, inputs.shape[1]], "sigmoid")

        # Error through epochs for plotting and early stopping
        self.losses = np.array([])  # every 1000 epochs
        self.patience = 10  # early stopping patience

        # Parametrize this to work for the denoising autoencoder.
        # The normal one will use the same input as output.
        self.expected_output = expected_output if expected_output is not None else self.inputs


    def train(self, epochs: int):
        for epoch in range(1, epochs):
            # Forward pass
            H_enc, V_enc, hO_enc, latent_vector = self.encoder.feed_forward(self.inputs)
            H_dec, V_dec, hO_dec, O = self.decoder.feed_forward(latent_vector)

            # Backward pass
            error = self.loss_derivative(self.expected_output, O) # error = dE/dO, where E is the loss function (e.g. binary cross entropy or MSE)

            delta_sum = self.decoder.backward_propagation(epoch, H_dec, V_dec, hO_dec, error)
            self.encoder.backward_propagation(epoch, H_enc, V_enc, hO_enc, delta_sum)
            
            if epoch % 1000 == 0:
                loss = self.loss(self.expected_output, O)
                self.losses = np.append(self.losses, loss)
                if settings.verbose: print(f"{epoch=} ; error={loss}\n")
                if self.early_stopping(): break


    def early_stopping(self, threshold: float = 0.01) -> bool:
        """Check if the loss didn't decrease by at least 1%."""
        if self.losses.shape[0] < 2:
            return False
        early_stop = self.losses[-1] > self.losses[-2] * (1 - threshold)
        if early_stop:
            self.patience -= 1
        return self.patience == 0


    def predict(self, inputs: np.array):
        """Predict the output for the given inputs."""
        _, _, _, latent_vector = self.encoder.feed_forward(inputs)
        _, _, _, O = self.decoder.feed_forward(latent_vector)
        return np.where(O < 0.5, 0, 1)  # threshold output to 0 or 1


    def loss(self, expected: np.array, actual: np.array):
        match settings.loss_function:
            case "bce":
                return self._binary_cross_entropy(expected, actual)
            case "mse":
                return self._mse(expected, actual)


    def loss_derivative(self, expected: np.array, actual: np.array):
        match settings.loss_function:
            case "bce":
                return self._binary_cross_entropy_derivative(expected, actual)
            case "mse":
                return self._mse_derivative(expected, actual)


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


    def visualize_loss(self):
        """Visualize the loss through epochs and save it to a file."""
        plt.plot(self.losses, marker="o", markersize=4)
        plt.yscale("log")
        plt.xlabel("Epochs (thousands)")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(settings.Config.output_path + "/autoencoder_loss.png")
        plt.close()


    def visualize_latent_space(self, labels):
        """Visualize the latent space and save it to a file."""
        sns.set_palette(sns.color_palette("viridis"))
        sns.set_theme(style="white")
        sns.set(font_scale=5, rc={"figure.figsize": (20, 20)}, style="whitegrid")
        _, _, _, latent_vector = self.encoder.feed_forward(self.inputs)
        ax = sns.scatterplot(
            x=latent_vector[:, 0],
            y=latent_vector[:, 1],
            marker="o",
            s=500,
            legend=False,
            color="red"
        )
    
        for i, (x, y) in enumerate(zip(latent_vector[:, 0], latent_vector[:, 1])):
            plt.text(x, y+1.15, labels[i], fontsize=24, ha='right', va='top')

        ax.invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(settings.Config.output_path + "/latent_space.png")
        plt.close()