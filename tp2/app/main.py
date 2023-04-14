import numpy as np
import math
import matplotlib.pyplot as plt

from settings import settings
from colors import mix_cmyk_colors, display_cmyk_colors
from selection import elite_selection, roulette_selection
from crossover import one_point_crossover, two_point_crossover, anular_crossover, uniform_crossover
from mutation import limited_mutation, uniform_mutation, complete_mutation

def main() -> None:
  # Load settings
  color_palette = tuple(settings.color_palette)
  target_color = tuple(settings.target_color)
  individuals_amt = settings.algorithm.individuals
  selection_method = settings.algorithm.selection_method
  display_interval = settings.visualization.display_interval
  crossover_method = settings.algorithm.crossover_method
  mutation_method = settings.algorithm.mutation_method

  # Initialize random population. Each individual is a 1D array of proportions for each color in the palette.
  population = init_population(individuals_amt, len(color_palette))

  # Get the result color rectangle to be updated during the genetic algorithm visualization
  result_color_rect = display_cmyk_colors(color_palette, (0,0,0,0), target_color)

  # Run genetic algorithm
  for iteration in range(10_000): # TODO: add stopping condition
    print(f"{iteration=}")

    # TODO: Crossover
    children = crossover(crossover_method, population)

    # TODO: Mutation
    children = mutation(mutation_method, children)

    # Calculate fitness for each individual in the population
    fitnesses = get_fitnesses(population, color_palette, target_color)

    population = selection(selection_method, population, fitnesses, individuals_amt)

    # Display the best individual of the current population
    if iteration % display_interval == 0:
      display_best_individual(result_color_rect, population, fitnesses, color_palette)


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


def display_best_individual(result_color_rect, population, fitnesses, color_palette) -> None:
  # Get the index of the individual with the highest fitness
  best_individual_idx = np.argmax(fitnesses)
  # Get the best individual
  best_individual = population[best_individual_idx]
  # Mix the colors together with the given proportions of the best individual
  result_color = mix_cmyk_colors(color_palette, best_individual)

  # Display the best individual
  try:
    result_color_rect.set_facecolor(result_color)
    plt.pause(0.0001)
  except ValueError:
    # If the proportions are invalid, the color will be invalid
    raise ValueError(f"Invalid color: {result_color}")
  

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


if __name__ == "__main__":
  main()
