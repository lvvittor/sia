import numpy as np

from settings import settings
from utils import parse_characters
from autoencoder import Autoencoder
import seaborn as sns
import matplotlib.pyplot as plt

def exercise_1():
    inputs = parse_characters(f"{settings.Config.data_path}/font.txt")
        
    autoencoder = Autoencoder(inputs, 16, 2)

    autoencoder.train(settings.epochs)
    
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
    step = (1 - settings.denoising_autoencoder.noise) / settings.denoising_autoencoder.train_iterations
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

            print(f"Finished iteration {iteration + 1} of {settings.denoising_autoencoder.train_iterations} with MSE {noise_iteration_mses[-1]} and noise {noise}")

        mses_by_noise_level.append(noise_iteration_mses)
    
    # Configurar el estilo y las paletas de color de Matplotlib
    plt.rcParams.update({
        "font.size": 50,
        "axes.labelsize": 50,
        "axes.titlesize": 60,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40,
        "legend.fontsize": 40
    })

    # Crear el gráfico de barras con Matplotlib
    fig, ax = plt.subplots(figsize=(30, 30))  # Ajusta el tamaño de la figura según tus necesidades
    ax.boxplot(mses_by_noise_level)
    ax.set_xticklabels([f"{noise:.2f}" for noise in noises])

    plt.title("MSE between the original and the denoised outputs")
    plt.xlabel("Noise level")
    plt.ylabel("MSE")
    
    plt.grid(True)
    plt.savefig(settings.Config.output_path + "/distances.png")
    plt.show()

    # print(f"Distance of the vectors: {np.mean(distances)}")



if __name__ == "__main__":
	match settings.exercise:
		case 1:
			exercise_1()
		case _:
			raise ValueError(f"Invalid exercise number: {settings.exercise}")