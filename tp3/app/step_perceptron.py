import numpy as np
from perceptron import Perceptron
from settings import settings
import seaborn as sns
import matplotlib.pyplot as plt


class StepPerceptron(Perceptron):
    def activation_func(self, value):
        return 1 if value >= 0 else -1  # step function

    def update_weights(self):
        # Get the difference between the expected outputs and the actual outputs
        output_errors = self.expected_outputs - self.get_outputs()

        # Compute the delta weights for each input
        deltas = self.learning_rate * output_errors.reshape(-1, 1) * self.inputs

        # Sum the delta weights for each input, and add them to the weights
        self.weights = self.weights + np.sum(deltas, axis=0)

    def get_error(self):
        return np.sum(abs(self.expected_outputs - self.get_outputs()))

    def is_converged(self):
        return self.get_error() == settings.step_perceptron.convergence_threshold

    def visualize(self):
        # # remove bias term
        drawable_inputs = self.inputs[:, 1:]

        sns.set_style("whitegrid")

        # plot the points
        sns.scatterplot(
            x=drawable_inputs[:, 0],
            y=drawable_inputs[:, 1],
            hue=self.get_outputs(),
            size=100,
            palette=["red", "blue"],
        )

        xmin, xmax = np.min(drawable_inputs[:, 0]), np.max(drawable_inputs[:, 0])

        # w1*x + w2*y + w0 = 0
        # y = -(w1*x + w0) / w2

        x = np.linspace(xmin - 100, xmax + 100, 1000)
        y = -(self.weights[1] * x + self.weights[0]) / self.weights[2]

        lineplot = sns.lineplot(x=x, y=y, color="black")

        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.show()

        # save the plot to a file
        fig = lineplot.get_figure()
        fig.savefig(f"{settings.Config.output_path}/step_perceptron.png")
