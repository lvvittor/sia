import numpy as np

def one_point_crossover(parent1, parent2, point: int = None):
    """One point crossover between two parents.

    Args:
        parent1 (list): A list of proportions for each color in the palette.
        parent2 (list): A list of proportions for each color in the palette.
        point (int): The index of the point to perform the crossover. If None, a random point is selected.

    Returns:
        A tuple of two children.
    """
    if point is None:
        point = np.random.randint(len(parent1))

    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))

    child1 = fix_incorrect_crossover(child1)
    child2 = fix_incorrect_crossover(child2)

    return child1, child2


def two_point_crossover(parent1, parent2, points: tuple[int, int] = None):
    """Two point crossover between two parents.

    Args:
        parent1 (list): A list of proportions for each color in the palette.
        parent2 (list): A list of proportions for each color in the palette.
        points (tuple[int, int]): The indices of the points to perform the crossover. If None, two random points are selected.

    Returns:
        A tuple of two children.
    """
    if points is None:
        points = np.random.randint(0, len(parent1), size=2)
        points.sort()

    child1 = np.concatenate((parent1[:points[0]], parent2[points[0]:points[1]], parent1[points[1]:]))
    child2 = np.concatenate((parent2[:points[0]], parent1[points[0]:points[1]], parent2[points[1]:]))

    child1 = fix_incorrect_crossover(child1)
    child2 = fix_incorrect_crossover(child2)

    return child1, child2


def anular_crossover(parent1, parent2, point: int = None, length: int = None):
    """Anular crossover between two parents.

    Args:
        parent1 (list): A list of proportions for each color in the palette.
        parent2 (list): A list of proportions for each color in the palette.
        point (int): The index of the point to perform the crossover. If None, a random point is selected.
        length (int): The length of the segment to perform the crossover. If None, a random length is selected.

    Returns:
        A tuple of two children.
    """
    if point is None:
        point = np.random.randint(len(parent1))

    if length is None:
        length = np.random.randint(1, np.ceil(len(parent1) / 2))

    child1 = np.concatenate((parent1[:point], parent2[point:point+length], parent1[point+length:]))
    child2 = np.concatenate((parent2[:point], parent1[point:point+length], parent2[point+length:]))

    if point + length > len(parent1):
        offset = len(parent1) - point
        length = length - offset
        child1 = np.concatenate((parent2[:length], child1[length:]))
        child2 = np.concatenate((parent1[:length], child2[length:]))

    child1 = fix_incorrect_crossover(child1)
    child2 = fix_incorrect_crossover(child2)

    return child1, child2


def uniform_crossover(parent1, parent2, probability: float = 0.5):
    """Uniform crossover between two parents.

    Args:
        parent1 (list): A list of proportions for each color in the palette.
        parent2 (list): A list of proportions for each color in the palette.
        probability (float): The probability of each gene to be inherited from the first parent.

    Returns:
        A tuple of two children.
    """
    mask = np.random.random(len(parent1)) < probability

    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)

    child1 = fix_incorrect_crossover(child1)
    child2 = fix_incorrect_crossover(child2)

    return child1, child2


def fix_incorrect_crossover(child):
    """Fixes a child that has incorrect proportions.

    Args:
        child (list): A list of proportions for each color in the palette.

    Returns:
        A list of proportions for each color in the palette.
    """
    proportion_sum = np.sum(child)

    if proportion_sum != 1:
        if proportion_sum == 0:
            child = np.ones(len(child)) / len(child)
        else:
            child = child / proportion_sum

    return child
