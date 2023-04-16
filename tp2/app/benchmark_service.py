import numpy as np
import json
import matplotlib.pyplot as plt
import math

from datetime import datetime
from utils import init_population, crossover, selection, get_fitness, mutation, sanity_check, timeout, get_population_fitness, get_best_color
from settings import settings

LABELS = {
    # "elite_one_point_limited": "Elite, one point and limited",
    # "elite_two_point_limited": "Elite, two point and limited",
    # "elite_anular_limited": "Elite, anular and limited",
    # "elite_uniform_limited": "Elite, uniform and limited",
    # "roulette_one_point_limited": "Roulette, one point and limited",
    # "roulette_two_point_limited": "Roulette, two point and limited",
    # "roulette_anular_limited": "Roulette, anular and limited",
    # "roulette_uniform_limited": "Roulette, uniform and limited",
    # "universal_one_point_limited": "Universal, one point and limited",
    # "universal_two_point_limited": "Universal, two point and limited",
    # "universal_anular_limited": "Universal, anular and limited",
    # "universal_uniform_limited": "Universal, uniform and limited",
    "ranking_one_point_limited": "Ranking, one point and limited",
    "ranking_two_point_limited": "Ranking, two point and limited",
    "ranking_anular_limited": "Ranking, anular and limited",
    "ranking_uniform_limited": "Ranking, uniform and limited",
    # "elite_one_point_uniform": "Elite, one point and uniform",
    # "elite_two_point_uniform": "Elite, two point and uniform",
    # "elite_anular_uniform": "Elite, anular and uniform",
    # "elite_uniform_uniform": "Elite, uniform and uniform",
    # "roulette_one_point_uniform": "Roulette, one point and uniform",
    # "roulette_two_point_uniform": "Roulette, two point and uniform",
    # "roulette_anular_uniform": "Roulette, anular and uniform",
    # "roulette_uniform_uniform": "Roulette, uniform and uniform",
    # "universal_one_point_uniform": "Universal, one point and uniform",
    # "universal_two_point_uniform": "Universal, two point and uniform",
    # "universal_anular_uniform": "Universal, anular and uniform",
    # "universal_uniform_uniform": "Universal, uniform and uniform",
    "ranking_one_point_uniform": "Ranking, one point and uniform",
    "ranking_two_point_uniform": "Ranking, two point and uniform",
    "ranking_anular_uniform": "Ranking, anular and uniform",
    "ranking_uniform_uniform": "Ranking, uniform and uniform",
    # "elite_one_point_complete": "Elite, one point and complete",
    # "elite_two_point_complete": "Elite, two point and complete",
    # "elite_anular_complete": "Elite, anular and complete",
    # "elite_uniform_complete": "Elite, uniform and complete",
    # "roulette_one_point_complete": "Roulette, one point and complete",
    # "roulette_two_point_complete": "Roulette, two point and complete",
    # "roulette_anular_complete": "Roulette, anular and complete",
    # "roulette_uniform_complete": "Roulette, uniform and complete",
    # "universal_one_point_complete": "Universal, one point and complete",
    # "universal_two_point_complete": "Universal, two point and complete",
    # "universal_anular_complete": "Universal, anular and complete",
    # "universal_uniform_complete": "Universal, uniform and complete",
    "ranking_one_point_complete": "Ranking, one point and complete",
    "ranking_two_point_complete": "Ranking, two point and complete",
    "ranking_anular_complete": "Ranking, anular and complete",
    "ranking_uniform_complete": "Ranking, uniform and complete",
}

class BenchmarkService:
  def __init__(self, color_palette, target_color, times, individuals):
    self.color_palette = color_palette
    self.target_color = target_color
    self.times = times
    self.individuals = individuals


  def plot_time_comparing_graph(self, benchmark):
      fig = plt.figure(figsize=(10, 5))

      mean_time = []
      std_time = []
      for key in benchmark.keys():
          mean_time.append(benchmark[key]["times"]["mean"])
          std_time.append(benchmark[key]["times"]["std"])

      xaxis = np.arange(len(LABELS))
      plt.xticks(xaxis, LABELS.values(), rotation=45)
      plt.bar(xaxis, mean_time, 0.4 ,yerr=std_time, align='center', alpha=0.5, ecolor='black', capsize=10, color="blue")
      plt.xlabel("Combinations")
      plt.ylabel('Time(s)')
      plt.title(f"Excecution Time")
      plt.grid(axis="y")

      # Save the figure and show
      plt.tight_layout()
      plt.savefig(f"{settings.Config.output_path}/time_comparation.png")
      plt.close()

  @timeout(seconds=settings.constraints.max_seconds)
  def make_experiment(self, selection_method, crossover_method, mutation_method, population):
    start_time = datetime.now()
    no_change_counter = 0
    previous_best_fitness = np.max(get_population_fitness(population))
    for _ in range(settings.constraints.max_generations):
      children = crossover(crossover_method, population) 

      sanity_check(children, "crossover")

      # Mutate children
      children = mutation(mutation_method, children)

      sanity_check(children, "mutation")

      # Add the children to the population
      population = np.concatenate((population, children))

      # Calculate fitness for each individual in the population
      fitnesses = get_population_fitness(population)

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

      population = selection(selection_method, population, fitnesses, self.individuals)
      sanity_check(population, "selection")
    end_time = datetime.now()

    return {
      "time": (end_time - start_time).total_seconds(),
      "best_candidate": population[np.argmax(fitnesses)]
    }


  def get_benchmark(self):
    results = {}
    counter = 0
    for key in LABELS.keys():
      results[key] = {
        "times": [],
        "best_candidates": []
      }
    
    # N poblaciones distintas
    # Para cada poblacion, corremos el algoritmo N veces
    for _ in range(self.times):
      population = init_population(self.individuals, len(self.color_palette))
      for algorithm in LABELS.keys():
        times = []
        best_candidates = []
        selection_method, crossover_method, mutation_method = [algorithm.split("_", 1)[0], algorithm.split("_", 1)[1].rsplit("_", 1)[0], algorithm.split("_", 1)[1].rsplit("_", 1)[1]]
        for _ in range(self.times):
          result = self.make_experiment(selection_method, crossover_method, mutation_method, population)
          times.append(result["time"])
          best_candidates.append(result["best_candidate"])
        results[algorithm]["times"].append(np.mean(times))
        # results[algorithm]["best_candidates"].append(best_candidates)
        
        print(f"Round {counter} ended: {algorithm}")
        counter += 1
        

    for algorithm in results.keys():
      results[algorithm] = { 
        "times": {
          "mean": np.mean(results[algorithm]["times"]),
          "std": np.std(results[algorithm]["times"])
        }
      }
      # TODO: que hacer con best_candidates
    
    filename = f"{settings.Config.output_path}/benchmark-data.json"
    with open(filename, "w") as file:
      file.write(json.dumps(results, default=lambda o: o.__dict__, indent=4))

    return results
