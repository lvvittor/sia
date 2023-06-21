import numpy as np

from settings import settings
from utils import parse_characters, visualize_character
from autoencoder import Autoencoder
from variational_autoencoder import VariationalAutoencoder

def exercise_1():
    inputs = parse_characters(f"{settings.Config.data_path}/font.txt")

    autoencoder = Autoencoder(inputs, [16], 2)

    autoencoder.train(settings.epochs)
    
	# Check if the error is of at most 1 pixel per character
    O = autoencoder.predict(inputs)

    incorrect_inputs = 0
    for i in range(inputs.shape[0]):
        if np.sum(np.abs(inputs[i] - O[i])) > 1:
            incorrect_inputs += 1
	
    # This MUST output 0 incorrect inputs
    print(f"Incorrect inputs: {incorrect_inputs} (out of {inputs.shape[0]})")
    print(f"Patience: {autoencoder.patience}")


def exercise_2():
    inputs = parse_characters(f"{settings.Config.data_path}/font.txt")

    vae = VariationalAutoencoder(inputs, [16], 2)

    vae.train(settings.epochs)

    latent_space = vae.predict_latent(inputs)
    labels = ['`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'DEL']
    vae.visualize_latent_space(latent_space, labels)
    vae.visualize_all_digits()

    O = vae.predict([[-0.25, -1.2], [-0.6, 0.5], [1, -0.3], [0.25, 0.8]])
    visualize_character(O[0], "characterB")
    visualize_character(O[1], "characterG")
    visualize_character(O[2], "characterO")
    visualize_character(O[3], "characterP")
	
    print(f"Finished training with patience: {vae.patience}")


if __name__ == "__main__":
    match settings.exercise:
        case 1:
            exercise_1()
        case 2:
            exercise_2()
        case _:
            raise ValueError(f"Invalid exercise number: {settings.exercise}")