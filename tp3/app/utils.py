import csv
import numpy as np


def get_boolean_value(value: int):
    return True if value == 1 else False


def logical_and(x: list[int, int]):
    return 1 if get_boolean_value(x[0]) and get_boolean_value(x[1]) else -1


def logical_xor(x: list[int, int]):
    return 1 if get_boolean_value(x[0]) ^ get_boolean_value(x[1]) else -1


def parse_csv(path: str, input_size: int = 3):
    with open(path, newline="") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        next(csv_reader)

        inputs = []
        expected_outputs = []

        for row in csv_reader:
            inputs.append(row[:input_size])
            expected_outputs.append(row[-1])

        # Return as numpy array of float numbers
        return np.array(inputs, dtype=float), np.array(expected_outputs, dtype=float)


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


if __name__ == "__main__":
    original_interval = (-1, 1)
    scaled_interval = (2, 30)

    value = 0.5
    scaled_value = feature_scaling(value, original_interval, scaled_interval)

    print(f"Scaled value {value} to {scaled_value}")
