from datetime import datetime
from algorithms import DFS, BFS, AStar, Greedy
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from settings import settings

ALGORITHMS = {
    "dfs": DFS, 
    "bfs": BFS, 
    "a_star": AStar, 
    "greedy":Greedy
}
HEURISTICS = ["farthest_region_heuristic", "distinct_colors", "composite"]
LABELS = {
    "dfs": "DFS",
    "bfs": "BFS",
    "a_star_farthest_region_heuristic": "A*(h1)",
    "a_star_distinct_colors": "A*(h2)",
    "a_star_composite": "A*(h3)",
    "a_star_cells_outside_zone": "A*(h4)",
    "greedy_farthest_region_heuristic": "Greedy(h1)",
    "greedy_distinct_colors": "Greedy(h2)",
    "greedy_composite": "Greedy(h3)",
    "greedy_cells_outside_zone": "Greedy(h4)",
}

class BenchMarkService:
    """
    Get the average and standard deviation execution time of a model run method using an input data set.
    """

    def __init__(self, times):
        self.times = times


    def plot_steps_comparing_graph(self, benchmark):
        fig = plt.figure(figsize=(10, 5))

        mean_costs = []
        std_costs = []
        for key in benchmark.keys():
            mean_costs.append(benchmark[key]["costs"]["mean"])
            std_costs.append(benchmark[key]["costs"]["std"])

        xaxis = np.arange(len(LABELS))
        plt.bar(xaxis, mean_costs, 0.4, yerr=std_costs, capsize=10)

        plt.xticks(xaxis, LABELS.values(), rotation=45)
        plt.xlabel("Algorithms")
        plt.ylabel("Solution Cost")
        plt.title(f"Solution costs for board {settings.board.N}x{settings.board.N} with {settings.board.M} colors")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(f"{settings.Config.output_path}/cost_comparation{settings.board.N}x{settings.board.N}c{settings.board.M}.png")
        plt.close()


    def plot_node_comparing_graph(self, benchmark):
        fig = plt.figure(figsize=(10, 5))

        expanded_nodes_mean = []
        expanded_nodes_std = []
        border_nodes_mean = []
        border_nodes_std = []
        for key in benchmark.keys():
            expanded_nodes_mean.append(benchmark[key]["expanded_nodes"]["mean"])
            expanded_nodes_std.append(benchmark[key]["expanded_nodes"]["std"])
            border_nodes_mean.append(benchmark[key]["border_nodes"]["mean"])
            border_nodes_std.append(benchmark[key]["border_nodes"]["std"])

        xaxis = np.arange(len(LABELS))
        plt.bar(xaxis - 0.2, expanded_nodes_mean, 0.4, label="Expanded nodes", capsize=10, yerr=expanded_nodes_std)
        plt.bar(xaxis + 0.2, border_nodes_mean, 0.4, label="Border nodes", capsize=10, yerr=border_nodes_std)

        plt.xticks(xaxis, LABELS.values(), rotation=45)
        plt.xlabel("Algorithms")
        plt.ylabel("Nodes")
        plt.title(f"Expanded and Border Nodes for board {settings.board.N}x{settings.board.N} with {settings.board.M} colors")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(f"{settings.Config.output_path}/node_comparation{settings.board.N}x{settings.board.N}c{settings.board.M}.png")
        plt.close()


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
        plt.xlabel("Algorithms")
        plt.ylabel('Time(s)')
        plt.title(f"Excecution Time for {settings.board.N}x{settings.board.N} with {settings.board.M} colors")
        plt.grid(axis="y")

        # Save the figure and show
        plt.tight_layout()
        plt.savefig(f"{settings.Config.output_path}/time_comparation{settings.board.N}x{settings.board.N}c{settings.board.M}.png")
        plt.close()

    def make_experiment(self, state, algorithm, heuristic=None):
        if heuristic:
            solver = ALGORITHMS[algorithm](state, heuristic)
        else:
            solver = ALGORITHMS[algorithm](state)

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
        # Corremos cada algoritmo, N veces para cada tablero distinto
        for _ in range(self.times):
            # Generamos un nuevo tablero
            state = board_generator.generate()
            for algorithm in LABELS.keys():
                times = []
                for _ in range(self.times):
                    state_copy = state.copy()
                    # Resolvemos el mismo tablero, con todos los algoritmos
                    if "greedy" in algorithm or "a_star" in algorithm:
                        # substract the `greedy_` or `a_star_` part from the algorithm name
                        heuristic = algorithm[7:]
                        _algorithm = algorithm[:6]
                        result = self.make_experiment(state_copy, _algorithm, heuristic)
                        results[algorithm]["heuristic"] = heuristic
                    else:
                        result = self.make_experiment(state_copy, algorithm)
                    times.append(result["time"])
                    
                
                results[algorithm]["times"].append(np.mean(times))
                results[algorithm]["costs"].append(result["cost"])
                results[algorithm]["expanded_nodes"].append(result["expanded"])
                results[algorithm]["border_nodes"].append(result["border"])

                print(f"Round {counter} ended: {algorithm}")
                counter += 1
        
        # Calcular mean y std para todos los algoritmos, una vez resueltos todos los mapas distintos
        for algorithm in results.keys():
            for medition in results[algorithm].keys():
                if medition == "heuristic":
                    continue
                mean = np.mean(results[algorithm][medition])
                std = np.std(results[algorithm][medition])
                results[algorithm][medition] = {"mean": mean, "std": std}


        filename = f"{settings.Config.output_path}/benchmark-data.json"
        with open(filename, "w") as file:
            file.write(json.dumps(results, default=lambda o: o.__dict__, indent=4))
        return results
