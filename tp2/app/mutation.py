import numpy as np
from settings import settings

def limited_mutation(individual):
    """Selects a random amount of genes, to mutate all of them with probability mutation_rate.

    Args:
        individual (list): A list of proportions for each color in the palette.

    Returns:
        The mutated individual.
    """

    if not (np.random.random() < settings.algorithm.mutation_rate):
        return individual
    
    gene_amt = np.random.randint(1, len(individual))

    # The locus are chosen randomly
    indexes = np.random.choice(len(individual), gene_amt, replace=False)

    # Grab pairs (g1, g2) of genes, add the delta to g1 and substract it from g2.
    for i in range(0, len(indexes), 2):
        g1 = indexes[i]
        g2 = indexes[i + 1]
        delta = np.random.uniform(-settings.algorithm.mutation_delta, settings.algorithm.mutation_delta)

        # Make sure the genes are not negative nor above 1
        if delta < 0:
            delta = -1 * min(abs(delta), individual[g1], 1-individual[g2])
        else:
            delta = min(delta, 1-individual[g1], individual[g2])

        individual[g1] += delta
        individual[g2] -= delta

    return individual


def complete_mutation(individual):
    """Mutation of an individual. With probability mutation_rate, all of the genes are mutated.

    Args:
        individual (list): A list of proportions for each color in the palette.

    Returns:
        The mutated individual.
    """

    if not (np.random.random() < settings.algorithm.mutation_rate):
        return individual

    # Grab pairs (g1, g2) of genes, add the delta to g1 and substract it from g2.
    for i in range(0, len(individual), 2):
        g1 = i
        g2 = i + 1
        delta = np.random.uniform(-settings.algorithm.mutation_delta, settings.algorithm.mutation_delta)

        # Make sure the genes are not negative nor above 1
        if delta < 0:
            delta = -1 * min(abs(delta), individual[g1], 1-individual[g2])
        else:
            delta = min(delta, 1-individual[g1], individual[g2])

        individual[g1] += delta
        individual[g2] -= delta

    return individual


def uniform_mutation(individual):
    """Mutation of an individual. Each PAIR of genes has a mutation_rate probability of being mutated.

    Args:
        individual (list): A list of proportions for each color in the palette.

    Returns:
        The mutated individual.
    """

    # Grab pairs (g1, g2) of genes, add the delta to g1 and substract it from g2.
    for i in range(0, len(individual), 2):
        if not (np.random.random() < settings.algorithm.mutation_rate):
            continue

        g1 = i
        g2 = i + 1
        delta = np.random.uniform(-settings.algorithm.mutation_delta, settings.algorithm.mutation_delta)

        # Make sure the genes are not negative nor above 1
        if delta < 0:
            delta = -1 * min(abs(delta), individual[g1], 1-individual[g2])
        else:
            delta = min(delta, 1-individual[g1], individual[g2])

        individual[g1] += delta
        individual[g2] -= delta

    return individual
