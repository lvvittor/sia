import numpy as np
import matplotlib.pyplot as plt
from settings import settings


def visualize_2d(X, Y, weights: list[float, float] = []):
    # Create a 2D scatter plot
    plt.scatter(X, Y, c="r", marker="o")

    # Set axis labels
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot the line
    if len(weights) > 0:
        line = weights[1] * X + weights[0]
        plt.plot(X, line, c="b")

    plt.savefig(f"{settings.Config.output_path}/visualize2d.png")


def visualize_3d(inputs):
    # Extract x, y, z coordinates from inputs
    x1 = []
    x2 = []
    x3 = []

    for data in inputs:
        x1.append(data[0])
        x2.append(data[1])
        x3.append(data[2])

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1, x2, x3, c="r", marker="o")

    # # Create a 2D plane
    # xx, yy = np.meshgrid(range(10), range(10))
    # zz = xx + yy  # Replace this with your own 2D plane equation
    # ax.plot_surface(xx, yy, zz, alpha=0.5)

    # Set axis labels
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

    # Show the plot
    plt.show()


def visualize_cross_validation():
    blocks = np.array([[0.7142, 0.5714, 0.2857, 0.2857], [0.4285, 0.8571, 0.1428, 0.8571], [0, 0.4285, 0.1428, 0.5714], [0.8571, 0.5714, 0.4285, 0.4285], [0.4285, 0.5714, 0.5714, 0.4285]])

    # Transponer la matriz de bloques
    blocks_T = np.transpose(blocks)

    # Configurar el gráfico
    fig, ax = plt.subplots()
    width = 0.2
    x = np.arange(5)

    # Iterar sobre cada bloque y crear una barra para cada elemento
    for i, element in enumerate(blocks_T):
        ax.bar(x + i * width + 0.1, element, width, label=f'Block {i+1}')

    # Agregar etiquetas y leyenda al gráfico
    ax.set_xticks(x + 2 * width + 0.1)
    ax.set_xticklabels(['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5'])
    ax.set_ylabel('Accuracy')
    ax.legend()

    # Mostrar el gráfico
    plt.show()
    plt.savefig(f"{settings.Config.output_path}/cross_validation.png")


def visualize_errors():
    # Definir los arreglos de valores para las dos líneas
    test_output = np.array([25.05310213, 7.23375041, 21.56969344, 17.90886506, 64.55373287, 1.13264102, 62.17524978])
    test_expected_output = np.array([24.974,  7.871, 21.755, 18.543, 64.107,  0.32, 61.301])

    # Configurar el gráfico
    fig, ax = plt.subplots()

    # Graficar la primera línea
    # ax.plot(test_output, label='Actual', color='blue')
    ax.scatter(np.arange(len(test_output)), test_output, label='Actual', color='blue')

    # Graficar la segunda línea
    # ax.plot(test_expected_output, label='Expected', color='red')
    ax.scatter(np.arange(len(test_expected_output)), test_expected_output, label='Expected', color='red', alpha=0.5)

    # Agregar etiquetas y leyenda al gráfico
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$Y^\mu$')
    ax.legend()

    # Mostrar el gráfico
    plt.show()
    plt.grid(alpha=0.5)
    plt.savefig(f"{settings.Config.output_path}/train_test_errors.png")


def visualize_digit_output():
    output = np.array([3.75921292e-02, 5.07557209e-04, 2.25636235e-03, 7.47481886e-01, 1.50909998e-03, 7.04594803e-03, 8.80117940e-02, 5.61648024e-03, 1.46905315e-03, 1.02530703e-01])

    # Configurar el gráfico
    fig, ax = plt.subplots()

    # Graficar la primera línea
    ax.scatter(np.arange(len(output)), output, color='blue')
    ax.set_xticks(np.arange(10))

    # Agregar etiquetas y leyenda al gráfico
    ax.set_xlabel('i')
    ax.set_ylabel('$O_{i}$')

    # Mostrar el gráfico
    plt.show()
    plt.grid(alpha=0.5)
    plt.savefig(f"{settings.Config.output_path}/digit_output.png")