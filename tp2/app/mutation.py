import numpy as np
from settings import settings

def limited_mutation(individual, amt):
    """Mutation of amt amount of genes from an individual, with probability mutation_rate.

    Args:
        individual (list): A list of proportions for each color in the palette.
        mutation_rate (float): The probability of mutating the gene.
        amt (int): Amount of genes to mutate.

    Returns:
        The mutated individual.
    """

    indexes = np.random.choice(len(individual), amt, replace=False)
    
    accumulated = 0

    for i in indexes[:-1]:
        if np.random.random() < settings.algorithm.mutation_rate:
            delta = np.random.uniform(-settings.algorithm.mutation_delta, settings.algorithm.mutation_delta)
            individual[i] += delta
            accumulated += delta

    individual[indexes[-1]] -= accumulated
    
    return individual

def uniform_mutation(individual):
    """Mutation of an individual. Each gene has a mutation_rate probability of being mutated.

    Args:
        individual (list): A list of proportions for each color in the palette.
        mutation_rate (float): The probability of mutating the gene.

    Returns:
        The mutated individual.
    """

    accumulated = 0

    for i in range(len(individual)):
        if np.random.random() < settings.algorithm.mutation_rate:
            delta = np.random.uniform(-settings.algorithm.mutation_delta, settings.algorithm.mutation_delta)
            individual[i] += delta
            accumulated += delta

    index = np.random.randint(len(individual))
    individual[index] -= accumulated

    return individual

def complete_mutation(individual):
    """Mutation of an individual. With probability mutation_rate, all of the genes are mutated.

    Args:
        individual (list): A list of proportions for each color in the palette.
        mutation_rate (float): The probability of mutating the gene.

    Returns:
        The mutated individual.
    """

    if np.random.random() < settings.algorithm.mutation_rate:
        accumulated = 0
        for i in range(len(individual)) - 1:
            delta = np.random.uniform(-settings.algorithm.mutation_delta, settings.algorithm.mutation_delta)
            individual[i] += delta
            accumulated += delta
        individual[-1] -= accumulated

    return individual


