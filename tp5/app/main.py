import numpy as np

from settings import settings
from utils import parse_characters
from autoencoder import Autoencoder

def exercise_1():
    inputs = parse_characters(f"{settings.Config.data_path}/font.txt")

    autoencoder = Autoencoder(inputs, 16, 2)

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


if __name__ == "__main__":
	match settings.exercise:
		case 1:
			exercise_1()
		case _:
			raise ValueError(f"Invalid exercise number: {settings.exercise}")