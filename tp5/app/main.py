from settings import settings
from utils import parse_characters, visualize_character

def exercise_1():
    inputs = parse_characters(f"{settings.Config.data_path}/font.txt")

    visualize_character(inputs[1])


if __name__ == "__main__":
	match settings.exercise:
		case 1:
			exercise_1()
		case _:
			raise ValueError("Invalid exercise number")