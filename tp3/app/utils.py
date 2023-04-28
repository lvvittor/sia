import csv
import numpy as np

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
        inputs.append(row[:3])
        expected_outputs.append(row[-1])

    # Return as numpy array of float numbers
    return np.array(inputs, dtype=float), np.array(expected_outputs, dtype=float)