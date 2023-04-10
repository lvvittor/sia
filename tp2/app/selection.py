def elite_selection(population, fitness, k):
  """Selects the k best individuals from the population.

  Args:
    population (list): A list of individuals.
    fitness (list): A list of fitness values.
    k (int): The number of individuals to select.

  Returns:
    A list of the k best individuals.
  """
  return [individual for _, individual in sorted(zip(fitness, population), key=lambda pair: pair[0], reverse=True)[:k]]
