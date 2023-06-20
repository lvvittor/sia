import numpy as np

from settings import settings
from utils import parse_characters, visualize_character
from autoencoder import Autoencoder
import seaborn as sns
import matplotlib.pyplot as plt

def exercise_1():
    inputs = parse_characters(f"{settings.Config.data_path}/font.txt")

    autoencoder = Autoencoder(inputs, 16, 2)

    autoencoder.train(settings.epochs)

    if settings.latent_space_points_to_add > 0:
        middle_points = np.zeros((settings.latent_space_points_to_add, 2))
        for i in range(settings.latent_space_points_to_add):
            indices = np.random.randint(0, len(autoencoder.latent_vector), size=2)
            p1, p2 = autoencoder.latent_vector[indices]
            middle_points[i] = (p1 + p2) / 2
            # middle_points[i] = (np.random.uniform(-10, 10), np.random.uniform(-5, 5))

            if settings.verbose:
                print(f"Added {middle_points[i]} to the latent space")
        
        modified_latent_vector = np.concatenate((autoencoder.latent_vector, middle_points))

        if settings.verbose:
            print(f"Modified latent space is: {modified_latent_vector}")

        _, _, _, O = autoencoder.decoder.feed_forward(modified_latent_vector)

        for i, o in enumerate(O):
            visualize_character(o, title=f"{i}")

    O = autoencoder.predict(inputs)

    # Check if the error is of at most 1 pixel per character
    incorrect_inputs = np.sum(np.sum(np.abs(inputs - O), axis=1) > 1)

    if settings.verbose:
        # This MUST output 0 incorrect inputs if we don't add noise to the inputs
        print(f"Incorrect inputs: {incorrect_inputs} (out of {inputs.shape[0]})")
        print(f"Patience: {autoencoder.patience}")

        # Visualize the latent space and save it to a file
        autoencoder.visualize_latent_space()

    # Apply some noise to the inputs (only if it's configured to add noise),
    # so the autoencoder can learn to denoise (i.e Denoising Autoencoder)
    if settings.denoising_autoencoder.noise > 0:
        _denoising_autoencoder(inputs)

def _denoising_autoencoder(inputs: np.array):
    """
    Create a denoising autoencoder and train it by adding noise the given inputs.
    """
    # Generate a list of noise levels to add to the inputs
    step = (
        1 - settings.denoising_autoencoder.noise
    ) / settings.denoising_autoencoder.train_iterations
    noises = np.arange(settings.denoising_autoencoder.noise, 1, step)

    # Mean Square error for the original and the denoised outputs
    mses_by_noise_level = []
    for iteration in range(settings.denoising_autoencoder.train_iterations):
        noise_iteration_mses = []
        for noise in noises:
            # Train the autoencoder with the original inputs
            autoencoder = Autoencoder(inputs, 16, 2)
            autoencoder.train(settings.epochs)
            O = autoencoder.predict(inputs)

            # Generate Gaussian noise with the desired noise level
            noise = np.random.normal(loc=0, scale=noise, size=inputs.shape)
            noisy_inputs = inputs + noise

            autoencoder = Autoencoder(noisy_inputs, 16, 2)
            autoencoder.train(settings.epochs)
            noisy_O = autoencoder.predict(noisy_inputs)

            # Mean Square error for the original and the denoised outputs
            noise_iteration_mses.append(np.mean(np.square(O - noisy_O)))

            if settings.verbose:
                print(
                    f"Finished iteration {iteration + 1} of {settings.denoising_autoencoder.train_iterations} with MSE {noise_iteration_mses[-1]} and noise {noise}"
                )

        mses_by_noise_level.append(noise_iteration_mses)

    # Plot the MSEs for each noise level
    plt.rcParams.update(
        {
            "font.size": 50,
            "axes.labelsize": 50,
            "axes.titlesize": 60,
            "xtick.labelsize": 40,
            "ytick.labelsize": 40,
            "legend.fontsize": 40,
        }
    )

    fig, ax = plt.subplots(figsize=(30, 30))
    ax.boxplot(mses_by_noise_level)
    ax.set_xticklabels([f"{noise:.2f}" for noise in noises])

    plt.title("MSE between the original and the denoised outputs")
    plt.xlabel("Noise level")
    plt.ylabel("MSE")

    plt.grid(True)
    plt.savefig(settings.Config.output_path + "/mses.png")
    plt.show()


if __name__ == "__main__":
    match settings.exercise:
        case 1:
            exercise_1()
        case _:
            raise ValueError(f"Invalid exercise number: {settings.exercise}")
