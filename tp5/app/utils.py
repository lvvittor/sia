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


def visualize_character(character: np.array, filename: str = "character"):
    sns.set(font_scale=5, rc={"figure.figsize": (20, 20)}, style="whitegrid")
    plt.figure()
    sns.heatmap(character.reshape(7, 5), cmap='Greys', vmin=0, vmax=1)
    plt.savefig(f"{settings.Config.output_path}/{filename}.png")
    plt.close()