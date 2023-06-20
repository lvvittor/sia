import numpy as np

from settings import settings
from utils import parse_characters
from autoencoder import Autoencoder

def exercise_1():
    inputs = parse_characters(f"{settings.Config.data_path}/font.txt")

    # Apply some noise to the inputs (only if it's configured to add noise), 
    # so the autoencoder can learn to denoise (i.e Denoising Autoencoder)
    if settings.noise > 0:
        # Generate Gaussian noise with the desired noise level
        noise = np.random.normal(loc=0, scale=settings.noise, size=inputs.shape)
        inputs = inputs + noise

    autoencoder = Autoencoder(inputs, 16, 2)

    autoencoder.train(settings.epochs)
    
	# Check if the error is of at most 1 pixel per character
    O = autoencoder.predict(inputs)

    print(f"Latent vector: {len(autoencoder.latent_vector)}")
    autoencoder.visualize_latent_space()

    incorrect_inputs = 0
    for i in range(inputs.shape[0]):
        if np.sum(np.abs(inputs[i] - O[i])) > 1:
            incorrect_inputs += 1
	
    # This MUST output 0 incorrect inputs if we don't add noise to the inputs
    print(f"Incorrect inputs: {incorrect_inputs} (out of {inputs.shape[0]})")
    print(f"Patience: {autoencoder.patience}")


if __name__ == "__main__":
	match settings.exercise:
		case 1:
			exercise_1()
		case _:
			raise ValueError(f"Invalid exercise number: {settings.exercise}")