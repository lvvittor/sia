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


def roulette_selection(population, fitness, k):
  # Normalize the fitness values (relative fitness)
  fitness = fitness / np.sum(fitness)

  # Create a list of cumulative fitness values
  cumulative_fitness = np.cumsum(fitness)

  selected = []

  for _ in range(k):
    # Select a random number between 0 and 1
    r = np.random.random()

    # Find the index of the first individual with a cumulative fitness value greater than r
    for idx, cumulative in enumerate(cumulative_fitness):
      if cumulative > r:
        selected.append(population[idx])
        break

  return selected
