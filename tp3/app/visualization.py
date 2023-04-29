import matplotlib.pyplot as plt

def visualize_2d(X, Y, weights: list[float, float] = []):
  # Create a 2D scatter plot
  plt.scatter(X, Y, c='r', marker='o')

  # Set axis labels
  plt.xlabel('x')
  plt.ylabel('y')

  # Plot the line
  if len(weights) > 0:
    line = weights[1] * X + weights[0]
    plt.plot(X, line, c='b')

  # Show the plot
  plt.show()


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
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x1, x2, x3, c='r', marker='o')

  # # Create a 2D plane
  # xx, yy = np.meshgrid(range(10), range(10))
  # zz = xx + yy  # Replace this with your own 2D plane equation
  # ax.plot_surface(xx, yy, zz, alpha=0.5)

  # Set axis labels
  ax.set_xlabel('x1')
  ax.set_ylabel('x2')
  ax.set_zlabel('x3')

  # Show the plot
  plt.show()