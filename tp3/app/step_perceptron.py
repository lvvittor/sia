import numpy as np
from perceptron import Perceptron
from settings import settings
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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

    def save_animation_frames(self):
        # remove bias term
        _inputs = self.inputs[:, 1:]

        for i, (weights, outputs) in enumerate(
            zip(self.historical_weights, self.historical_outputs)
        ):
            # plot the points
            sns.scatterplot(
                x=_inputs[:, 0],
                y=_inputs[:, 1],
                hue=outputs,
                style=outputs,
                palette=["red", "blue"],
                marker="x",
            )

            xmin, xmax = np.min(_inputs[:, 0]), np.max(_inputs[:, 0])
            x = np.linspace(xmin - 100, xmax + 100, 1000)

            # w1*x + w2*y + w0 = 0 => y = -(w1*x + w0) / w2

            # w1*x + w2*y + w0 = 0 => y = -(w1*x + w0) / w2
            if weights[2] == 0:
                y = np.zeros(len(x))
            else:
                y = -(weights[1] * x + weights[0]) / weights[2]

            lineplot = sns.lineplot(x=x, y=y, color="black")

            plt.xlim([-2, 2])
            plt.ylim([-2, 2])
            plt.legend(markerscale=2)
            plt.title(f"Step Perceptron Epoch {i}")

            # save the plot to a file
            fig = lineplot.get_figure()
            fig.savefig(f"{settings.Config.output_path}/step_perceptron_{i}.png")

            # clear the current figure to prevent overlapping of plots
            plt.clf()

    def save_animation(self):
        # remove bias term
        _inputs = self.inputs[:, 1:]

        fig, ax = plt.subplots()

        def update(i):
            ax.clear()

            weights, outputs = self.historical_weights[i], self.historical_outputs[i]

            # plot the points
            sns.scatterplot(
                x=_inputs[:, 0],
                y=_inputs[:, 1],
                hue=outputs,
                style=outputs,
                palette=["red", "blue"],
                marker="x",
            )

            xmin, xmax = np.min(_inputs[:, 0]), np.max(_inputs[:, 0])
            x = np.linspace(xmin - 100, xmax + 100, 1000)

            # w1*x + w2*y + w0 = 0 => y = -(w1*x + w0) / w2
            if weights[2] == 0:
                y = np.zeros(len(x))
            else:
                y = -(weights[1] * x + weights[0]) / weights[2]

            # plot the separating hyperplane
            ax.plot(x, y, c="k")

            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_title(f"Step Perceptron Epoch {i}")

        anim = FuncAnimation(
            fig, update, frames=len(self.historical_weights), interval=500
        )

        anim.save(
            f"{settings.Config.output_path}/step_perceptron.gif", writer="imagemagick"
        )

        fig.clf()
