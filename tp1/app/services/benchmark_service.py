from datetime import datetime
from algorithms import DFS, BFS, AStar, Greedy
import pandas as pd
import numpy as np
import json
import copy
import matplotlib.pyplot as plt
from settings import settings

ALGORITHMS = {"dfs": DFS, "bfs": BFS, "a_star": AStar, "greedy":Greedy}
HEURISTICS = ["farthest_region_heuristic", "distinct_colors", "composite"]

class BenchMarkService:
    """
    Get the average and standard deviation execution time of a model run method using an input data set.
    """

    def __init__(self, state, times):
        self.state = state
        self.times = times


    def plot_box_graph(benchmark):
        pass


    def plot_time_comparing_graph(self, benchmark):
        # TODO: verify figsize
        fig = plt.figure(figsize=(10, 5))

        mean_time = []
        std_time = []
        for key in benchmark.keys():
            mean_time.append(benchmark[key]["mean"])
            std_time.append(benchmark[key]["std"])

        fig, ax = plt.subplots()
        ax.bar(benchmark.keys(), mean_time, yerr=std_time, align='center', alpha=0.5, ecolor='black', capsize=10, color="blue")
        ax.set_ylabel('Time(s)')
        ax.set_xticks(mean_time)
        ax.set_xticklabels(benchmark.keys())
        ax.set_title(f"Excecution Time for {settings.board.N}x{settings.board.N}")
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig(f"{settings.Config.output_path}/time_comparation{settings.board.N}x{settings.board.N}.png")

    def make_experiment(self, data, algorithm, heuristic=None):

        current_data = copy.deepcopy(data)
        for _ in range(self.times):
            aux_state = self.state.copy()
            if heuristic:
                solver = ALGORITHMS[algorithm](aux_state, heuristic)
            else:
                solver = ALGORITHMS[algorithm](aux_state)
            start_time = datetime.now()
            # TODO: change this to include expanded and frontier in results
            # _, cost, expanded, frontier = solver.solve()
            _, cost = solver.solve()
            end_time = datetime.now()

            current_data["times"].append((end_time - start_time).total_seconds())
            current_data["cost"].append(cost)
        current_data["mean"] = np.mean(current_data["times"])
        current_data["std"] = np.std(current_data["times"])

        return current_data

    def get_benchmark(self):
        """
        Run the benchmark and get the average execution time.

        Also, gets the standard deviation of the execution time.
        """

        # We should add to this dictionaty the respective classes for the algorithms
        data = {
            "times":[],
            "mean": 0,
            "std": 0,
            "cost": []
        }
        results = {}
        for algorithm in ALGORITHMS.keys():

            if algorithm in ["greedy", "a_star"]:
                for heuristic in HEURISTICS:
                    current_data = self.make_experiment(data, algorithm, heuristic)
                    current_data.update({"heuristic": heuristic})
                    results.update({f"{algorithm}_{heuristic}": current_data})
            else:
                current_data = self.make_experiment(data, algorithm)
                results.update({algorithm: current_data})

        filename = f"{settings.Config.output_path}/benchmark-data.json"
        with open(filename, "w") as file:
            file.write(json.dumps(results))
        return results
