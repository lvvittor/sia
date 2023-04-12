import numpy as np
import math
import matplotlib.pyplot as plt

from settings import settings
from colors import mix_cmyk_colors, display_cmyk_colors
from selection import elite_selection, roulette_selection

from utils import timeout

@timeout(seconds=settings.constraints.max_seconds)
def run_genetic_algorithm(population, result_color_rect) -> bool:
  for generation in range(settings.constraints.max_generations):
    print(f"{generation=}")

    # TODO: Crossover

    # TODO: Mutation

    # Calculate fitness for each individual in the population
    fitnesses = get_fitnesses(population, settings.color_palette, settings.target_color)

    population = selection(settings.algorithm.selection_method, population, fitnesses, settings.algorithm.individuals)

    # Display the best individual of the current population
    if generation % settings.visualization.display_interval == 0:
      display_best_individual(result_color_rect, population, fitnesses, settings.color_palette)
  
  return True





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
    mixed_color = mix_cmyk_colors(color_palette, individual)

    # Calculate the fitness for each individual
    fitnesses[i] = get_fitness_of_color(mixed_color, target_color)

  return fitnesses


def get_fitness_of_color(color: tuple, target_color: tuple) -> float:
  """Calculate the fitness of a color"""
  # Max distance between any two CMYK colors (is a constant of number 2)
  max_dist = math.dist((0, 0, 0, 0), (1, 1, 1, 1)) 
  
  # Compute Euclidean distance between the two colors
  dist = math.dist(color, target_color)

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
    # TODO: add other selection methods
    case _:
      raise ValueError(f"Invalid selection method: {selection_method}")


def stop_condition():
  """
  Returns True if the genetic algorithm should stop, False otherwise.
  """
  return False

if __name__ == "__main__":
  # Initialize random population. Each individual is a 1D array of proportions for each color in the palette.
  population = init_population(settings.algorithm.individuals, len(settings.color_palette))

  # Get the result color rectangle to be updated during the genetic algorithm visualization
  result_color_rect = display_cmyk_colors(settings.color_palette, (0,0,0,0), settings.target_color)
  
  if run_genetic_algorithm(population, result_color_rect):
    print("Genetic algorithm finished successfully")
  else:
    print("Genetic algorithm timed out")
