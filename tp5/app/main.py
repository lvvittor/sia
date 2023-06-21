import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from settings import settings
from utils import add_noise, parse_characters, visualize_character, binary_cross_entropy, visualize_characters
from autoencoder import Autoencoder
from variational_autoencoder import VariationalAutoencoder

def exercise_1():
    inputs = parse_characters(f"{settings.Config.data_path}/font.txt")

    autoencoder = Autoencoder(inputs, [16], 2)

    autoencoder.train(settings.epochs)

    if settings.middle_point.execute:    
        # Feed forward the encoder with the selected 2 inputs
        _, _, _, p = autoencoder.encoder.feed_forward(np.array([inputs[settings.middle_point.first_input_index], inputs[settings.middle_point.second_input_index]]))

        # Compute the middle point between the 2 points
        middle_point = np.array((p[0] + p[1]) / 2)
        _, _, _, decoded_p = autoencoder.decoder.feed_forward(np.array([p[0], p[1], middle_point]))

        if settings.verbose:
            print(f"Feed forward the encoder with the first 2 inputs: {inputs[:2]}")
            print(f"Output of the encoder: {p}")
            print(f"Point for the input of the decoder: {middle_point}")
            print(f"Output of the decoder: {decoded_p}")

        characters = [decoded_p[0], decoded_p[1], decoded_p[2]]
        titles = [f"Character {settings.middle_point.first_input_index + 1}", f"Character {settings.middle_point.second_input_index + 1}", f"Middle point"]
        visualize_characters(characters, suptitle="Middle point", titles=titles, filename=f"middle-point-{settings.middle_point.first_input_index}-{settings.middle_point.second_input_index}")

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
    if settings.denoising_autoencoder.execute:
        _denoising_autoencoder(inputs)

def _denoising_autoencoder(original_inputs: np.array) -> None:
    """Train a denoising autoencoder and plot the predicted denoised inputs MSEs for each noise level.
    
        Args:
            original_inputs: Original inputs to the autoencoder

        Returns:
            None
    """
        
    # Train the denoising autoencoder with noisy inputs and the original inputs as expected outputs)
    original_inputs_repeated = np.repeat(original_inputs, settings.denoising_autoencoder.data_augmentation_factor, axis=0)
    noisy_inputs = add_noise(original_inputs_repeated, noise_level=settings.denoising_autoencoder.train_noise)
    autoencoder = Autoencoder(inputs=noisy_inputs, hidden_nodes=[16], latent_dim=7, expected_output=original_inputs_repeated)
    autoencoder.train(settings.epochs)

    # Now that the autoencoder is trained, we can use it to denoise different noisy inputs

    binary_cross_entropies_by_noise_level = []
    for iteration in range(settings.denoising_autoencoder.predict_rounds):
        noise_iteration_binary_cross_entropies = []
        for noise in settings.denoising_autoencoder.predict_noises:
            
            # Predict the denoised inputs and compute the MSE between the original and the denoised inputs
            noisy_inputs = add_noise(original_inputs, noise_level=noise)
            predicted_denoised_inputs = autoencoder.predict(noisy_inputs)
            noise_iteration_binary_cross_entropies.append(binary_cross_entropy(original_inputs, predicted_denoised_inputs))

            if settings.verbose:
                print(
                    f"Finished iteration {iteration + 1} of {settings.denoising_autoencoder.predict_rounds} with Binary Cross Entropy {noise_iteration_binary_cross_entropies[-1]} and noise {noise}"
                )

            # Visualize the original and denoised inputs for the last iteration
            if iteration == settings.denoising_autoencoder.predict_rounds - 1:
                for i, (original_input, noisy_input, predicted_denoised_input) in enumerate(zip(original_inputs, noisy_inputs, predicted_denoised_inputs)):
                    if i < 5: # Only visualize the first 5 characters
                            characters = [original_input, noisy_input, predicted_denoised_input]
                            titles = [f"original", f"noisy:{noise}", f"denoised"]
                            visualize_characters(characters, titles, filename=f"iteration-{iteration}-noise-{noise}-character-{i}")


        binary_cross_entropies_by_noise_level.append(noise_iteration_binary_cross_entropies)

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
    ax.boxplot(binary_cross_entropies_by_noise_level)
    ax.set_xticklabels([f"{noise:.2f}" for noise in settings.denoising_autoencoder.predict_noises])

    plt.title("Binary Cross Entropy between the original and the denoised outputs")
    plt.xlabel("Noise level")
    plt.ylabel("Binary Cross Entropy")

    plt.grid(True)
    plt.savefig(settings.Config.output_path + "/binary-cross-entropy.png")
    plt.show()


def exercise_2():
    inputs = parse_characters(f"{settings.Config.data_path}/font.txt")

    vae = VariationalAutoencoder(inputs, [16], 2)

    vae.train(settings.epochs)

    latent_space = vae.predict_latent(inputs)
    labels = ['`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'DEL']
    vae.visualize_latent_space(latent_space, labels)
    vae.visualize_all_digits()

    O = vae.predict([[-0.25, -1.2], [-0.6, 0.5], [1, -0.3], [0.25, 0.8]])
    visualize_character(O[0], "blue")
    visualize_character(O[1], "green")
    visualize_character(O[2], "orange")
    visualize_character(O[3], "purple")
	
    print(f"Finished training with patience: {vae.patience}")


if __name__ == "__main__":
    match settings.exercise:
        case 1:
            exercise_1()
        case 2:
            exercise_2()
        case _:
            raise ValueError(f"Invalid exercise number: {settings.exercise}")
