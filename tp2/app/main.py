import numpy as np
import math
import matplotlib.pyplot as plt

from settings import settings
from colors import mix_cmyk_colors, display_cmyk_colors
from selection import elite_selection, ranking_selection, roulette_selection, universal_selection
from crossover import one_point_crossover, two_point_crossover, anular_crossover, uniform_crossover

from utils import timeout

@timeout(seconds=settings.constraints.max_seconds)
def run_genetic_algorithm(population, result_color_rect) -> bool:
  """Run the genetic algorithm"""
  no_change_counter = 0
  previous_best_fitness = np.max(get_population_fitness(population))
  for generation in range(settings.constraints.max_generations):
    print(f"{generation=}")

    # TODO: Crossover
    children = crossover(population)

    # TODO: Mutation

    # Calculate fitness for each individual in the population
    fitnesses = get_population_fitness(population)

    population = selection(population, fitnesses)

    current_best_fitness = np.max(fitnesses)

    # Check if the best fitness doesn't change for the next 10 generations
    if math.isclose(current_best_fitness, previous_best_fitness, rel_tol=1e-9, abs_tol=0.0):
      no_change_counter += 1
    else:
      no_change_counter = 0
      previous_best_fitness = current_best_fitness

    if no_change_counter >= settings.constraints.acceptable_fitness_stagnation:
      break
    
    # The acceptable fitness is reached, stop the algorithm (TODO: maybe compare floats with math.isclose)
    if math.isclose(get_fitness(get_best_color(population, fitnesses), settings.target_color), settings.constraints.acceptable_fitness, rel_tol=1e-9, abs_tol=0.0):
      break

    # Display the best individual of the current population
    if generation % settings.visualization.display_interval == 0:
      display_best_individual(result_color_rect, population, fitnesses)
  
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

def get_best_color(population, fitnesses):
  # Get the index of the individual with the highest fitness
  best_individual_idx = np.argmax(fitnesses)
  
  # Get the best individual
  best_individual = population[best_individual_idx]

  # Mix the colors together with the given proportions of the best individual
  mixed_color = mix_cmyk_colors(settings.color_palette, best_individual)

  return mixed_color

def display_best_individual(result_color_rect, population, fitnesses) -> None:
  best_color = get_best_color(population, fitnesses)

  # Display the best individual
  try:
    result_color_rect.set_facecolor(best_color)
    plt.pause(0.0001)
  except ValueError:
    # If the proportions are invalid, the color will be invalid
    raise ValueError(f"Invalid color: {best_color}")

def selection(population: list[list[float]], fitnesses: list[float]):
  """Select the k best individuals from the population"""
  match settings.algorithm.selection_method:
    case "elite":
      return elite_selection(population, fitnesses, k=settings.algorithm.individuals)
    case "roulette":
      return roulette_selection(population, fitnesses, k=settings.algorithm.individuals)
    case "universal":
      return universal_selection(population, fitnesses, k=settings.algorithm.individuals)
    case "ranking":
      return ranking_selection(population, fitnesses, k=settings.algorithm.individuals)
    case _:
      raise ValueError(f"Invalid selection method: {settings.algorithm.selection_method}")

def crossover(population: list[list[float]]):
  """Crossover the population"""
  children = []

  for i in range(0, len(population), 2):
    match settings.algorithm.crossover_method:
      case "one_point":
        children.extend(one_point_crossover(population[i], population[i+1]))
      case "two_point":
        children.extend(two_point_crossover(population[i], population[i+1]))
      case "anular":
        children.extend(anular_crossover(population[i], population[i+1]))
      case "uniform":
        children.extend(uniform_crossover(population[i], population[i+1]))
      case _:
        raise ValueError(f"Invalid crossover method: {settings.algorithm.crossover_method}")
  
  return children


if __name__ == "__main__":
  # Initialize random population. Each individual is a 1D array of proportions for each color in the palette.
  population = init_population(settings.algorithm.individuals, len(settings.color_palette))

  # Get the result color rectangle to be updated during the genetic algorithm visualization
  result_color_rect = display_cmyk_colors(settings.color_palette, (0,0,0,0), settings.target_color)
  
  if run_genetic_algorithm(population, result_color_rect):
    print("Genetic algorithm finished successfully")
  else:
    print("Genetic algorithm timed out")
