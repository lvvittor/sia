import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from settings import settings

def get_boolean_value(value: int):
    return True if value == 1 else False


def logical_and(x: list[int, int]):
    return 1 if get_boolean_value(x[0]) and get_boolean_value(x[1]) else -1


def logical_xor(x: list[int, int]):
    return 1 if get_boolean_value(x[0]) ^ get_boolean_value(x[1]) else -1


def parse_csv(path: str):
  with open(path, newline='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    next(csv_reader)

    inputs = []
    expected_outputs = []

    for row in csv_reader:
        inputs.append(row[:-1])
        expected_outputs.append(row[-1])

    # Return as numpy array of float numbers
    return np.array(inputs, dtype=float), np.array(expected_outputs, dtype=float)
  

def parse_digits(path: str):
    with open(path, 'r') as f:
        data = f.read().splitlines()  # read the file and split into lines

    digit_images = []
    
    for i, line in enumerate(data):
        if i % 7 == 0:  # if we're at the start of a new item
            digit_img = []  # create a new digit_img array
        numbers = line.strip().split()  # remove trailing spaces and split into individual numbers
        digit_img.extend(numbers)  # add the numbers to the current digit_img
        if i % 7 == 6:  # if we're at the end of an digit_img
            digit_images.append(digit_img)  # add the digit_img to the list of digit_images

    digits = np.arange(10)

    return np.array(digit_images, dtype=float), digits


def train_test_split(*arrays, test_size=0.25, random_state=None):
    """
    Split arrays or matrices into random train and test subsets
    """
    n_samples = arrays[0].shape[0]
    indices = np.arange(n_samples)
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(indices)
    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    train_arrays = tuple(
        [
            array[train_indices]
            if isinstance(array, np.ndarray)
            else [array[i] for i in train_indices]
            for array in arrays
        ]
    )
    test_arrays = tuple(
        [
            array[test_indices]
            if isinstance(array, np.ndarray)
            else [array[i] for i in test_indices]
            for array in arrays
        ]
    )
    return *train_arrays, *test_arrays


def feature_scaling(
    value: float, from_int: tuple[float, float], to_int: tuple[float, float]
) -> float:
    numerator = value - from_int[0]
    denominator = from_int[1] - from_int[0]
    return (numerator / denominator) * (to_int[1] - to_int[0]) + to_int[0]


def visualize_digit(digit: np.array):
	sns.heatmap(digit.reshape(7, 5), cmap='Greys', vmin=0, vmax=1)

	# Mostrar el heatmap
	plt.show()
	plt.savefig(f"{settings.Config.output_path}/digit.png")


def visualize_error(data):

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set axis limits
    ax.set_xlim(0, len(data))
    ax.set_ylim(0, max(point[1] for point in data))

    # Create empty lists to store x and y values
    x_values = []
    y_values = []

    # Iterate over data and plot each point with a red dot
    for point in data:
        x_values.append(point[0])
        y_values.append(point[1])
        ax.plot(point[0], point[1], 'ro')

        # Plot a line connecting the points
        if len(x_values) > 1:
            ax.plot(x_values, y_values, 'b--')

    # Save the plot as a PNG image
    plt.savefig(f"{settings.Config.output_path}/error_vs_epoch.png")



if __name__ == "__main__":

    # FEATURE SCALING TEST
    original_interval = (-1, 1)
    scaled_interval = (2, 30)

    value = 0.5
    scaled_value = feature_scaling(value, original_interval, scaled_interval)

    print(f"Scaled value {value} to {scaled_value}")
    

    # TRAIN TEST SPLIT TEST
    X, y = parse_csv(f"../data/test_data.csv")

    print(X)
    print(y)

    # Split the data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # # Print the sizes of the training and testing subsets
    print(f"X_train: {X_train}")
    print(f"y_train: {y_train}")
    print(f"X_test: {X_test}")
    print(f"y_test: {y_test}")
