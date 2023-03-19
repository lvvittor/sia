from datetime import datetime
import numpy as np

class BenchMarkService:
    """
    Get the average and standard deviation execution time of a model run method using an input data set.
    """

    def __init__(self, model, data, times):
        self.model = model
        self.data = data
        self.times = times

    def get_benchmark(self):
        """
        Run the benchmark and get the average execution time.
        
        Also, gets the standard deviation of the execution time.
        """
        time_list = []
        for _ in range(self.times):
            start_time = datetime.now()
            self.model.run(self.data)
            end_time = datetime.now()
            time_list.append(end_time - start_time)

        return np.mean(time_list), np.std(time_list)
