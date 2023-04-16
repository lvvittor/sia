import numpy as np
import math
import matplotlib.pyplot as plt

from settings import settings
from colors import mix_cmyk_colors, display_cmyk_colors
from benchmark_service import BenchmarkService
from utils import init_population, crossover, selection, get_fitnesses, mutation, sanity_check

def main() -> None:
  # Load settings
  color_palette = tuple(settings.color_palette)
  target_color = tuple(settings.target_color)
  
  if settings.benchmarks.active == True:
    benchmarkService = BenchmarkService(color_palette, target_color, settings.benchmarks.rounds, settings.benchmarks.individuals)
    result = benchmarkService.get_benchmark()
    benchmarkService.plot_time_comparing_graph(result)
    return

  individuals_amt = settings.algorithm.individuals
  selection_method = settings.algorithm.selection_method
  display_interval = settings.visualization.display_interval
  crossover_method = settings.algorithm.crossover_method
  mutation_method = settings.algorithm.mutation_method

  # Initialize random population. Each individual is a 1D array of proportions for each color in the palette.
  population = init_population(individuals_amt, len(color_palette))

  sanity_check(population, "init_population")

  # Get the result color rectangle to be updated during the genetic algorithm visualization
  result_color_rect, result_text = display_cmyk_colors(color_palette, (0,0,0,0), target_color)

  # Run genetic algorithm
  for iteration in range(100_000): # TODO: add stopping condition
    print(f"{iteration=}")

    # Crossover current population
    children = crossover(crossover_method, population)

    sanity_check(children, "crossover")

    # Mutate children
    children = mutation(mutation_method, children)

    sanity_check(children, "mutation")

    # Add the children to the population
    population = np.concatenate((population, children))

    # Calculate fitness for each individual in the population
    fitnesses = get_fitnesses(population, color_palette, target_color)

    # Display the best individual of the current generation
    if iteration % display_interval == 0:
      display_best_individual(result_color_rect, result_text, population, fitnesses, color_palette)

    # Select the "best" individuals to form the next generation
    population = selection(selection_method, population, fitnesses, individuals_amt)

    sanity_check(population, "selection")


def display_best_individual(result_color_rect, result_text, population, fitnesses, color_palette) -> None:
  # Get the index of the individual with the highest fitness
  best_individual_idx = np.argmax(fitnesses)
  # Get the best individual
  best_individual = population[best_individual_idx]
  # Mix the colors together with the given proportions of the best individual
  result_color = mix_cmyk_colors(color_palette, best_individual)

  print(f"Best fitness: {np.max(fitnesses)}")

  # Display the best individual
  try:
    result_color_rect.set_facecolor([round(p, 5) for p in result_color])
    result_text.set_text(str(tuple(round(c, 2) for c in result_color)))
    plt.pause(0.0001)
  except ValueError:
    # If the proportions are invalid, the color will be invalid
    raise ValueError(f"{best_individual=} ; invalid_color={result_color}")

if __name__ == "__main__":
  main()
