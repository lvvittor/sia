import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from settings import settings

def parse_characters(path: str):
    with open(path, 'r') as f:
        data = f.read().splitlines()  # read the file and split into lines

    character_images = []
    
    for i, line in enumerate(data):
        if i % 7 == 0:  # if we're at the start of a new item
            char_img = []  # create a new char_img array
        numbers = line.strip().split()  # remove trailing spaces and split into individual numbers
        char_img.extend(numbers)  # add the numbers to the current char_img
        if i % 7 == 6:  # if we're at the end of an char_img
            character_images.append(char_img)  # add the char_img to the list of character_images

    return np.array(character_images, dtype=float)


def feature_scaling(
    value: float, from_int: tuple[float, float], to_int: tuple[float, float]
) -> float:
    numerator = value - from_int[0]
    denominator = from_int[1] - from_int[0]
    return (numerator / denominator) * (to_int[1] - to_int[0]) + to_int[0]

def add_noise(array: np.array, noise_level: float) -> np.array:
    """Add Gaussian noise to the given array.

    Args:
        array: Array to add noise to
        noise_level: Noise level in [0, 1]

    Returns:
        A new array with added noise
    """
    return np.clip(array + np.random.normal(loc=0, scale=noise_level, size=array.shape), 0, 1)

def mse(expected: np.array, actual: np.array) -> float:
    """Compute the Mean Squared Error between the expected and actual values.

    Args:
        expected: Expected values
        actual: Actual values

    Returns:
        The MSE between the expected and actual values
    """
    return np.mean(np.square(expected - actual))


def visualize_character(character: np.array, title: str = None):
    sns.set(font_scale=5, rc={"figure.figsize": (20, 20)}, style="whitegrid")
    plt.figure()
    sns.heatmap(character.reshape(7, 5), cmap='Greys', vmin=0, vmax=1)
    
    plt.savefig(f"{settings.Config.output_path}/character-{title}.png")
    plt.show()