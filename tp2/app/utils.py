import numpy as np
import math

from crossover import one_point_crossover, two_point_crossover, anular_crossover, uniform_crossover
from selection import elite_selection, roulette_selection, ranking_selection, universal_selection
from colors import mix_cmyk_colors

def init_population(individuals_amt: int, colors_amt: int):
  """Initialize a random population of individuals"""
  population = np.zeros((individuals_amt, colors_amt))

  for i in range(individuals_amt):
    # Make sure the proportions of the individual sum up to 1
    proportion_to_distribute = 1

    while proportion_to_distribute > 0:
      # Pick a random color from the palette
      color_idx = np.random.randint(colors_amt)
      # Pick a random proportion to distribute to the color
      proportion = np.random.random()

      if proportion > proportion_to_distribute:
        proportion = proportion_to_distribute

      # Update the proportion to distribute
      proportion_to_distribute -= proportion
      # Update the proportion of the color in the individual
      population[i][color_idx] += proportion

  return population


def selection(selection_method: str, population: list[list[float]], fitnesses: list[float], k: int):
  """Select the k best individuals from the population"""
  match selection_method:
    case "elite":
      return elite_selection(population, fitnesses, k)
    case "roulette":
      return roulette_selection(population, fitnesses, k)
    case "universal":
      return universal_selection(population, fitnesses, k)
    case "ranking":
      return ranking_selection(population, fitnesses, k)
    case _:
      raise ValueError(f"Invalid selection method: {selection_method}")

def crossover(crossover_method: str, population: list[list[float]]):
  """Crossover the population"""
  children = []

  for i in range(0, len(population), 2):
    match crossover_method:
      case "one_point":
        children.extend(one_point_crossover(population[i], population[i+1]))
      case "two_point":
        children.extend(two_point_crossover(population[i], population[i+1]))
      case "anular":
        children.extend(anular_crossover(population[i], population[i+1]))
      case "uniform":
        children.extend(uniform_crossover(population[i], population[i+1]))
      case _:
        raise ValueError(f"Invalid crossover method: {crossover_method}")
  
  return children



def get_fitnesses(population: np.ndarray, color_palette: list[tuple], target_color: tuple) -> np.ndarray[float]:
  """Calculate the fitness for each individual in the population"""
  fitnesses = np.zeros(len(population))

  for i, individual in enumerate(population):
    # Mix the colors together with the given proportions of the individual
    result_color = mix_cmyk_colors(color_palette, individual)
    # Calculate the fitness for each individual
    fitnesses[i] = fitness_fn(result_color, target_color)

  return fitnesses


def fitness_fn(color: tuple, target_color: tuple) -> float:
  """Calculate the fitness of a color"""
  max_dist = math.sqrt(1**2 + 1**2 + 1**2 + 1**2)  # max distance between any two CMYK colors

  # Compute Euclidean distance between the two colors
  dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(color, target_color)]))

  # Normalize distance to a range between 0 and 1
  fitness = 1 - (dist / max_dist)

  return fitness