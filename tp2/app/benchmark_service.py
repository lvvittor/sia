import numpy as np
import json
import matplotlib.pyplot as plt

from datetime import datetime
from utils import init_population, crossover, selection, get_fitnesses
from settings import settings

LABELS = {
    "elite_one_point": "Elite and one point",
    "elite_two_point": "Elite and two point",
    "elite_anular": "Elite and anular",
    "elite_uniform": "Elite and uniform",
    "roulette_one_point": "Roulette and one point",
    "roulette_two_point": "Roulette and two point",
    "roulette_anular": "Roulette and anular",
    "roulette_uniform": "Roulette and uniform",
    "universal_one_point": "Universal and one point",
    "universal_two_point": "Universal and two point",
    "universal_anular": "Universal and anular",
    "universal_uniform": "Universal and uniform",
    # "ranking_one_point": "Ranking and one point",
    # "ranking_two_point": "Ranking and two point",
    # "ranking_anular": "Ranking and anular",
    # "ranking_uniform": "Ranking and uniform",
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

  def make_experiment(self, selection_method, crossover_method, population):
    start_time = datetime.now()
    for i in range(100): # Condicion de corte
      children = crossover(crossover_method, population) 

      # TODO: Mutation

      fitnesses = get_fitnesses(population, self.color_palette, self.target_color)

      population = selection(selection_method, population, fitnesses, self.individuals)
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
        selection_method, crossover_method = algorithm.split("_", 1) 
        for _ in range(self.times):
          result = self.make_experiment(selection_method, crossover_method, population)
          times.append(result["time"])
          best_candidates.append(result["best_candidate"])
        results[algorithm]["times"].append(np.mean(times))
        # results[algorithm]["best_candidates"].append(best_candidates)
        
        print(f"Round {counter} ended: {algorithm}")
        counter += 1
        

    print(results)
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
