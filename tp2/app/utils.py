import numpy as np
import math
import func_timeout

from functools import wraps
from crossover import one_point_crossover, two_point_crossover, anular_crossover, uniform_crossover
from selection import elite_selection, roulette_selection, ranking_selection, universal_selection
from mutation import limited_mutation, uniform_mutation, complete_mutation
from colors import mix_cmyk_colors
from settings import settings

def timeout(seconds: int):
    """
    A decorator that times out a function after a given amount of seconds
    and returns False if the function times out.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func_timeout.func_timeout(
                    seconds, func, args=args, kwargs=kwargs
                )
            except func_timeout.FunctionTimedOut:
                return False

        return wrapper

    return decorator

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



def get_population_fitness(population: np.ndarray) -> np.ndarray[float]:
  """Calculate the fitness for each individual in the population"""
  fitnesses = np.zeros(len(population))

  for i, individual in enumerate(population):
    # Mix the colors together with the given proportions of the individual
    mixed_color = mix_cmyk_colors(settings.color_palette, individual)
    # Calculate the fitness for each individual
    fitnesses[i] = get_fitness(mixed_color,  settings.target_color)

  return fitnesses


def get_fitness(color: tuple, target_color: tuple) -> float:
  """Calculate the fitness of a color"""
  # Max distance between any two CMYK colors (is a constant of number 2)
  max_dist = math.dist((0, 0, 0, 0), (1, 1, 1, 1)) 

  # Compute Euclidean distance between the two colors
  dist = math.dist(color, target_color)

  # Normalize distance to a range between 0 and 1
  fitness = 1 - (dist / max_dist)

  return fitness
  

def sanity_check(population, step: str) -> None:
  for individual in population:
    if not math.isclose(sum(individual), 1):
      print(f"{individual=}")
      print(f"ERROR: individual proportions don't sum up to 1, at step **{step}**")
      exit(1)


def mutation(mutation_method: str, population: list[list[float]]):
  """Mutate each individual of the population"""
  for i in range(len(population)):
    match mutation_method:
      case "limited":
        population[i] = limited_mutation(population[i])
      case "uniform":
        population[i] = uniform_mutation(population[i])
      case "complete":
        population[i] = complete_mutation(population[i])
      case _:
        raise ValueError(f"Invalid mutation method: {mutation_method}")

  return population

def get_best_color(population, fitnesses):
  # Get the index of the individual with the highest fitness
  best_individual_idx = np.argmax(fitnesses)

  # Get the best individual
  best_individual = population[best_individual_idx]

  print(f"Best individual: {best_individual}")

  # Mix the colors together with the given proportions of the best individual
  mixed_color = mix_cmyk_colors(settings.color_palette, best_individual)

  return mixed_color
