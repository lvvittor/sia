import numpy as np

"""Functions to select `k` individuals from the population, according to a specific criteria.

  Args:
    population (list): A list of individuals.
    fitness (list): A list of fitness values.
    k (int): The number of individuals to select.

  Returns:
    A list of the k individuals selected.
"""

def elite_selection(population, fitness, k):
  return [individual for _, individual in sorted(zip(fitness, population), key=lambda pair: pair[0], reverse=True)[:k]]


def roulette_selection(population, fitness, k, universal=None):
  # Normalize the fitness values (relative fitness)
  fitness = fitness / np.sum(fitness)

  # Create a list of cumulative fitness values
  cumulative_fitness = np.cumsum(fitness)

  selected = []

  r = np.random.random()

  for i in range(k):
    # Select a random number between 0 and 1
    r = np.random.random() if universal is None else (r+i)/k

    # Find the index of the first individual with a cumulative fitness value greater than r
    for idx, cumulative in enumerate(cumulative_fitness):
      if cumulative > r:
        selected.append(population[idx])
        break

  return selected

def universal_selection(population, fitness, k):
  return roulette_selection(population, fitness, k, True)

def ranking_selection(population, fitness, k):

    ranked = sorted(fitness, reverse=True)
    selected = []

    for i in len(fitness):
        selected[i] = (len(fitness) - ranked.index(fitness[i]) + 1)/len(fitness)

    return roulette_selection(population, selected, k)
