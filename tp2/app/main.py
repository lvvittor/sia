import numpy as np
import math
import matplotlib.pyplot as plt

from settings import settings
from colors import display_cmyk_colors
from benchmark_service import BenchmarkService
from utils import init_population, crossover, selection, get_fitness, mutation, sanity_check, timeout, get_population_fitness, get_best_color

@timeout(seconds=settings.constraints.max_seconds)
def run_genetic_algorithm(population, result_color_rect, result_text) -> bool:

  no_change_counter = 0
  previous_best_fitness = np.max(get_population_fitness(population))

  # Run genetic algorithm
  for iteration in range(settings.constraints.max_generations):
    # Crossover current population
    children = crossover(settings.algorithm.crossover_method, population)

    sanity_check(children, "crossover")

    # Mutate children
    children = mutation(settings.algorithm.mutation_method, children)

    sanity_check(children, "mutation")

    # Add the children to the population
    population = np.concatenate((population, children))

    # Calculate fitness for each individual in the population
    fitnesses = get_population_fitness(population)

    current_best_fitness = np.max(fitnesses)

    print(f"{iteration=} ; current_best_fitness{round(current_best_fitness, 2)}")

    # Check if the best fitness doesn't change for the next 10 generations
    if math.isclose(current_best_fitness, previous_best_fitness, rel_tol=1e-9, abs_tol=0.0):
      no_change_counter += 1
    else:
      no_change_counter = 0
      previous_best_fitness = current_best_fitness

    if no_change_counter >= settings.constraints.acceptable_fitness_stagnation:
      break

    # The acceptable fitness is reached, stop the algorithm (TODO: maybe compare floats with math.isclose)
    if current_best_fitness >= settings.constraints.acceptable_fitness:
      break

    # Display the best individual of the current generation
    if iteration % settings.visualization.display_interval == 0:
      display_best_individual(result_color_rect, result_text, population, fitnesses, iteration)

    # Select the "best" individuals to form the next generation
    population = selection(settings.algorithm.selection_method, population, fitnesses, settings.algorithm.individuals)

    sanity_check(population, "selection")

  return True


def display_best_individual(result_color_rect, result_text, population, fitnesses, iteration) -> None:
  best_color = get_best_color(population, fitnesses)
  # Display the best individual
  try:
    display_cmyk_colors(color_palette, [round(p, 5) for p in best_color], target_color, iteration)
    # result_color_rect.set_facecolor([round(p, 5) for p in best_color])
    # result_text.set_text(str(tuple(round(c, 2) for c in best_color)))
    # plt.pause(0.0001)
  except ValueError:
    # If the proportions are invalid, the color will be invalid
    raise ValueError(f"invalid_color={best_color}")

if __name__ == "__main__":
  # Load settings
  color_palette = tuple(settings.color_palette)
  target_color = tuple(settings.target_color)
  
  if settings.benchmarks.active == True:
    benchmarkService = BenchmarkService(color_palette, target_color, settings.benchmarks.rounds, settings.benchmarks.individuals)
    result = benchmarkService.get_benchmark()
    benchmarkService.plot_time_comparing_graph(result)
    benchmarkService.plot_generation_comparing_graph(result)
    benchmarkService.plot_fitness_comparing_graph(result)
  else:
    # Initialize random population. Each individual is a 1D array of proportions for each color in the palette.
    population = init_population(settings.algorithm.individuals, len(color_palette))
    sanity_check(population, "init_population")

    # Get the result color rectangle to be updated during the genetic algorithm visualization
    result_color_rect, result_text = display_cmyk_colors(color_palette, (0,0,0,0), target_color, 0)
    if run_genetic_algorithm(population, result_color_rect, result_text):
      print("Genetic algorithm finished successfully")
    else:
      print("Genetic algorithm timed out")
