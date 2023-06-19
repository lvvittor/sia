from settings import settings
from utils import parse_characters
from autoencoder import Autoencoder

def exercise_1():
    inputs = parse_characters(f"{settings.Config.data_path}/font.txt")

    autoencoder = Autoencoder(inputs[:1], 12, 2)

    autoencoder.train(settings.epochs)


if __name__ == "__main__":
	match settings.exercise:
		case 1:
			exercise_1()
		case _:
			raise ValueError(f"Invalid exercise number: {settings.exercise}")