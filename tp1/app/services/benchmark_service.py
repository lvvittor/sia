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
    "bfs": "BFS",
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

    def make_experiment(self, algorithm, heuristic=None):
        aux_state = self.state.copy()
        if heuristic:
            solver = ALGORITHMS[algorithm](aux_state, heuristic)
        else:
            solver = ALGORITHMS[algorithm](aux_state)
        start_time = datetime.now()
        _, cost, expanded_nodes, border_nodes = solver.solve()
        end_time = datetime.now()

        return {
            "time": (end_time - start_time).total_seconds(),
            "cost": cost,
            "expanded": expanded_nodes,
            "border": border_nodes
        }

    def get_benchmark(self, board_generator):
        """
        Run the benchmark and get the average execution time.

        Also, gets the standard deviation of the execution time.
        """
        counter = 0

        results = {}

        for key in LABELS.keys():
            results[key] = {
                "times": [],
                "costs": [],
                "expanded_nodes": [],
                "border_nodes": []
            }


        # N tableros distintos
        # Corremos cada algoritmo, 1 vez para cada tablero distinto
        # No corremos el mismo algoritmo con el mismo tablero mas de una vez
        for _ in range(self.times):
            # Generamos un nuevo tablero
            self.state = board_generator.generate()
            for algorithm in LABELS.keys():
                # Resolvemos el mismo tablero, con todos los algoritmos
                if "greedy" in algorithm or "a_star" in algorithm:
                    # substract the `greedy_` or `a_star_` part from the algorithm name
                    heuristic = algorithm[7:]
                    _algorithm = algorithm[:6]         
                    result = self.make_experiment(_algorithm, heuristic)
                    results[algorithm]["heuristic"] = heuristic
                else:
                    result = self.make_experiment(algorithm)
                
                results[algorithm]["times"].append(result["time"])
                results[algorithm]["costs"].append(result["cost"])
                results[algorithm]["expanded_nodes"].append(result["expanded"])
                results[algorithm]["border_nodes"].append(result["border"])

                print(f"Round {counter} ended: {algorithm}")
                counter += 1
        
        # Calcular mean y std para todos los algoritmos, una vez resueltos todos los mapas distintos

        # Graficar

        filename = f"{settings.Config.output_path}/benchmark-data.json"
        with open(filename, "w") as file:
            file.write(json.dumps(results))
        return results
