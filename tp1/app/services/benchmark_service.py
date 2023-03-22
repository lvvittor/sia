from datetime import datetime
import numpy as np

class BenchMarkService:
    """
    Get the average and standard deviation execution time of a model run method using an input data set.
    """

    def __init__(self, board, times):
        self.board = board
        self.times = times


    def plot_box_graph(benchmark):
        pass


    def plot_time_comparing_graph(benchmark):
        pass



    def get_benchmark(self):
        """
        Run the benchmark and get the average execution time.

        Also, gets the standard deviation of the execution time.
        """

        # We should add to this dictionaty the respective classes for the algorithms
        algorithms = {
            "dfs": {
                "class":DFS,
                "times":[],
                "mean": 0,
                "std": 0,
                "cost": []
            }
        }

        for algorithm in algorithms.keys()
            for _ in range(self.times):
                solver = algorithms[algorithm]["class"](self.board)
                start_time = datetime.now()
                _, cost = solver.solve(self.board)
                end_time = datetime.now()
                algorithms[algorithm]["class"].append(end_time - start_time)
                algorithms[algorithm]["cost"].append(cost)
            algorithms[algorithm]["mean"] = np.mean(time_list)
            algorithms[algorithm]["std"] = np.std(time_list)

        return algorithms
