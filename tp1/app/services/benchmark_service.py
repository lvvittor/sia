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
LABELS = {
    "dfs": "DFS",
    "BFS": "BFS",
    "a_star_farthest_region_heuristic": "A*(h1)",
    "a_star_distinct_colors": "A*(h2)",
    "a_star_composite": "A*(h3)",
    "greedy_farthest_region_heuristic": "Greedy(h1)",
    "greedy_distinct_colors": "Greedy(h2)",
    "greedy_composite": "Greedy(h3)",
}

class BenchMarkService:
    """
    Get the average and standard deviation execution time of a model run method using an input data set.
    """

    def __init__(self, state, times):
        self.state = state
        self.times = times


    def plot_steps_comparing_graph(self, benchmark):
        fig = plt.figure(figsize=(10, 5))

        costs = []
        for key in benchmark.keys():
            costs.append(benchmark[key]["cost"])

        xaxis = np.arange(len(LABELS))
        plt.bar(xaxis, costs, 0.4)

        plt.xticks(xaxis, LABELS.values(), rotation=45)
        plt.xlabel("Algorithms")
        plt.ylabel("Nodes")
        plt.title(f"Expanded and Border Nodes for board {settings.board.N}x{settings.board.N}")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(f"{settings.Config.output_path}/node_comparation{settings.board.N}x{settings.board.N}.png")
        plt.close()


    def plot_node_comparing_graph(self, benchmark):
        fig = plt.figure(figsize=(10, 5))

        expanded_nodes = []
        border_nodes = []
        for key in benchmark.keys():
            expanded_nodes.append(np.mean(benchmark[key]["cost"]))

        xaxis = np.arange(len(LABELS))
        plt.bar(xaxis - 0.2, expanded_nodes, 0.4, label="Expanded nodes")
        plt.bar(xaxis + 0.2, border_nodes, 0.4, label="Border nodes")

        plt.xticks(xaxis, LABELS.values(), rotation=45)
        plt.xlabel("Algorithms")
        plt.ylabel("Nodes")
        plt.title(f"Expanded and Border Nodes for board {settings.board.N}x{settings.board.N}")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(f"{settings.Config.output_path}/node_comparation{settings.board.N}x{settings.board.N}.png")
        plt.close()


    def plot_time_comparing_graph(self, benchmark):
        fig = plt.figure(figsize=(10, 5))

        mean_time = []
        std_time = []
        for key in benchmark.keys():
            mean_time.append(benchmark[key]["mean"])
            std_time.append(benchmark[key]["std"])

        xaxis = np.arange(len(LABELS))
        plt.xticks(xaxis, LABELS.values(), rotation=45)
        plt.bar(xaxis, mean_time, 0.4 ,yerr=std_time, align='center', alpha=0.5, ecolor='black', capsize=10, color="blue")
        plt.xlabel("Algorithms")
        plt.ylabel('Time(s)')
        plt.title(f"Excecution Time for {settings.board.N}x{settings.board.N}")
        plt.grid(axis="y")

        # Save the figure and show
        plt.tight_layout()
        plt.savefig(f"{settings.Config.output_path}/time_comparation{settings.board.N}x{settings.board.N}.png")
        plt.close()

    def make_experiment(self, data, algorithm, heuristic=None):

        current_data = copy.deepcopy(data)
        for _ in range(self.times):
            aux_state = self.state.copy()
            if heuristic:
                solver = ALGORITHMS[algorithm](aux_state, heuristic)
            else:
                solver = ALGORITHMS[algorithm](aux_state)
            start_time = datetime.now()
            _, cost, expanded_nodes, border_nodes = solver.solve()Stashed changes
            end_time = datetime.now()

            current_data["times"].append((end_time - start_time).total_seconds())
            current_data["cost"].append(cost)
        current_data["mean"] = np.mean(current_data["times"])
        current_data["std"] = np.std(current_data["times"])
        current_data["expanded"] = expanded_nodes
        current_data["border"] = border_nodes

        return current_data

    def get_benchmark(self):
        """
        Run the benchmark and get the average execution time.

        Also, gets the standard deviation of the execution time.
        """

        data = {
            "times":[],
            "mean": 0,
            "std": 0,
            "cost": [],
            "expanded": 0,
            "border": 0
        }
        results = {}
        counter = 0
        for algorithm in ALGORITHMS.keys():

            if algorithm in ["greedy", "a_star"]:
                for heuristic in HEURISTICS:
                    current_data = self.make_experiment(data, algorithm, heuristic)
                    current_data.update({"heuristic": heuristic})
                    results.update({f"{algorithm}_{heuristic}": current_data})
                    print(f"Round {counter} ended: {algorithm} - {heuristic}")
                    counter += 1
            else:
                current_data = self.make_experiment(data, algorithm)
                results.update({algorithm: current_data})
                print(f"Round {counter} ended: {algorithm}")
                counter += 1

        filename = f"{settings.Config.output_path}/benchmark-data.json"
        with open(filename, "w") as file:
            file.write(json.dumps(results))
        return results
