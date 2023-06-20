import numpy as np

from settings import settings
from utils import add_noise, parse_characters, visualize_character, mse
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
        
        # modified_latent_vector = np.concatenate((autoencoder.latent_vector, middle_points))

        if settings.verbose:
            print(f"Modified latent space is: {middle_points}")

        _, _, _, O = autoencoder.decoder.feed_forward(middle_points)

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
    if settings.denoising_autoencoder.train_noise > 0:
        _denoising_autoencoder(inputs)

def _denoising_autoencoder(original_inputs: np.array) -> None:
    """Train a denoising autoencoder and plot the predicted denoised inputs MSEs for each noise level.
    
        Args:
            original_inputs: Original inputs to the autoencoder

        Returns:
            None
    """

    # Train the denoising autoencoder with noisy inputs and the original inputs as expected outputs)
    noisy_inputs = add_noise(original_inputs, noise_level=settings.denoising_autoencoder.train_noise)
    autoencoder = Autoencoder(inputs=noisy_inputs, hidden_nodes=16, latent_dim=2, expected_output=original_inputs)
    autoencoder.train(settings.epochs)

    # Now that the autoencoder is trained, we can use it to denoise different noisy inputs

    mses_by_noise_level = []
    for iteration in range(settings.denoising_autoencoder.predict_rounds):
        noise_iteration_mses = []
        for noise in settings.denoising_autoencoder.predict_noises:
            
            # Predict the denoised inputs and compute the MSE between the original and the denoised inputs
            noisy_inputs = add_noise(original_inputs, noise_level=noise)
            predicted_denoised_inputs = autoencoder.predict(noisy_inputs)
            noise_iteration_mses.append(mse(original_inputs, predicted_denoised_inputs))

            if settings.verbose:
                print(
                    f"Finished iteration {iteration + 1} of {settings.denoising_autoencoder.predict_rounds} with MSE {noise_iteration_mses[-1]} and noise {noise}"
                )
                # Visualize the original and denoised inputs for the last iteration
                if iteration == settings.denoising_autoencoder.predict_rounds - 1:
                    for i, (original_input, noisy_input, predicted_denoised_input) in enumerate(zip(original_inputs, noisy_inputs, predicted_denoised_inputs)):
                        if i < 5: # Only visualize the first 5 characters
                                # print(f"Visualizing character {i}")
                                # print(f"Original input: {original_input}")
                                # print(f"Noisy input: {noisy_input}")
                                # print(f"Predicted denoised input: {predicted_denoised_input}")
                            visualize_character(original_input, title=f"{i}-original-iteration-{iteration}")
                            visualize_character(noisy_input, title=f"{i}-noisy:{noise}-iteration-{iteration}")
                            visualize_character(predicted_denoised_input, title=f"{i}-denoised:{noise}-iteration-{iteration}")


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
    ax.set_xticklabels([f"{noise:.2f}" for noise in settings.denoising_autoencoder.predict_noises])

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
