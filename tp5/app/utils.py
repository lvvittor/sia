import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

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

def binary_cross_entropy(expected: np.array, actual: np.array, epsilon=1e-15):
    """Compute the binary cross entropy loss function."""
    P = np.clip(actual, epsilon, 1 - epsilon)  # avoid log(0)
    return np.mean(-expected * np.log(P) - (1 - expected) * np.log(1 - P))

def visualize_character(character: np.array, title: str = None):
    sns.set(font_scale=5, rc={"figure.figsize": (20, 20)}, style="whitegrid")
    plt.figure()
    sns.heatmap(character.reshape(7, 5), cmap='Greys', vmin=0, vmax=1)
    
    plt.savefig(f"{settings.Config.output_path}/character-{title}.png")
    plt.close()

def visualize_characters(characters: List[np.array], suptitle: str = "Denoising Autoencoder", titles: List[str] = None, filename: str = "characters.png"):
    sns.set(font_scale=15, rc={"figure.figsize": (100, 55)}, style="whitegrid")
    plt.figure()
    num_characters = len(characters)
    num_rows = int(np.ceil(num_characters / 3))
    num_cols = min(num_characters, 3)
    
    fig, axes = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(hspace=1)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    
    for i, character in enumerate(characters):
        row = i // num_cols
        col = i % num_cols
        
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.set_title(titles[i]) if titles else None
        sns.heatmap(character.reshape(7, 5), cmap='Greys', vmin=0, vmax=1, ax=ax, cbar=i == 0, cbar_ax=None if i else cbar_ax)
        ax.axis('off')
    
    if titles:
        fig.suptitle(suptitle)
    
    plt.savefig(f"{settings.Config.output_path}/characters-{filename}.png")
    plt.close()
