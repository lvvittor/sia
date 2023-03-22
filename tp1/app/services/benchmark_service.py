from datetime import datetime
from algorithms import DFS, BFS
import numpy as np
import matplotlib.pyplot as plt
from settings import settings

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
        plt.savefig(f"{settings.Config.output_path}/time_comparation2.png")


    def get_benchmark(self):
        """
        Run the benchmark and get the average execution time.

        Also, gets the standard deviation of the execution time.
        """

        # We should add to this dictionaty the respective classes for the algorithms
        algorithms = {
            "dfs": {
                "class": DFS,
                "times":[],
                "mean": 0,
                "std": 0,
                "cost": []
            },
            "bfs": {
                "class": BFS,
                "times":[],
                "mean": 0,
                "std": 0,
                "cost": []
            }
        }

        for algorithm in algorithms.keys():
            for _ in range(self.times):
                aux_state = self.state.copy()
                solver = algorithms[algorithm]["class"](aux_state)
                start_time = datetime.now()
                _, cost = solver.solve()
                end_time = datetime.now()
                algorithms[algorithm]["times"].append((end_time - start_time).total_seconds())
                algorithms[algorithm]["cost"].append(cost)
            algorithms[algorithm]["mean"] = np.mean(algorithms[algorithm]["times"])
            algorithms[algorithm]["std"] = np.std(algorithms[algorithm]["times"])

        return algorithms
