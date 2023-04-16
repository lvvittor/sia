import matplotlib.pyplot as plt

def mix_cmyk_colors(colors: list[tuple], proportions: list[float]) -> tuple[float, float, float, float]:
  """Mixes a list of CMYK colors together using the given proportions.

  Args:
    colors (list): A list of colors to mix together.
    proportions (list): A list of proportions for each color.

  Returns:
    tuple: A tuple of four floats representing the resulting color.
  """
  result_color = [0, 0, 0, 0]
  for i, color in enumerate(colors):
    for j in range(len(result_color)):
      result_color[j] += color[j] * proportions[i]
  return tuple(result_color)


def display_cmyk_colors(colors: list[tuple], result_color: tuple, target_color: tuple):
  """Displays a list of CMYK colors"""
  num_colors = len(colors)
  num_columns = min(num_colors, 4) # display up to 4 columns
  num_rows = (num_colors + num_columns - 1) // num_columns
  # Add an extra row to show the result and target colors
  _, axs = plt.subplots(num_rows + 1, num_columns, figsize=(2*num_columns, 2*num_rows))

  # Display each CMYK color as a rectangle on a subplot
  for i, color in enumerate(colors):
    row = i // num_columns
    col = i % num_columns
    # Hide the x and y axes
    axs[row, col].axes.get_xaxis().set_visible(False)
    axs[row, col].axes.get_yaxis().set_visible(False)
    # Add a rectangle to the subplot
    axs[row, col].add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color))
    axs[row, col].set_title("Color {}".format(i+1))

  # Hide any unused subplots
  for i in range(num_colors, num_rows*num_columns):
    axs.flat[i].set_visible(False)
  
  # Display the result and target colors in the middle of the last row
  result_col = num_columns // 2
  target_col = num_columns // 2 + 1
  for i in range(num_columns):
    # hide the numbers on the x axes
    axs[num_rows, i].tick_params(axis='x', labelbottom=False)
    # hide the y axes
    axs[num_rows, i].axes.get_yaxis().set_visible(False)
    if i == result_col:
      result_color_rect = plt.Rectangle((0, 0), 1, 1, facecolor=result_color)
      axs[num_rows, i].add_patch(result_color_rect)
      axs[num_rows, i].set_title("Best Approx.")
      # display the color tuple below the rectangle, rounded to 2 decimal places
      axs[num_rows, i].set_xlabel(str(tuple(round(c, 2) for c in result_color)))
    elif i == target_col:
      axs[num_rows, i].add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=target_color))
      axs[num_rows, i].set_title("Target")
      axs[num_rows, i].set_xlabel(target_color)
    else:
      axs[num_rows, i].set_visible(False)

  # Display the plot
  plt.show(block=False)

  # Return the rectangle to be updated
  return result_color_rect


# Test when calling as a script
if __name__ == "__main__":

  # Define the mixing proportions for each color
  proportions = [0.5, 0.5, 0, 0]

  # CMYK colors are represented as a tuple of four values, where each value is a float between 0 and 1
  colors = [
    (1, 0, 0, 1), # red
    (0, 0, 1, 1), # blue
    (0.5, 0.5, 1, 1),
    (0.5, 0.5, 1, 1)
  ]

  assert len(proportions) == len(colors), "The number of proportions must match the number of colors"

  target_color = (0.5, 0.5, 0, 1)

  # Mix the colors together with the given proportions
  result_color = mix_cmyk_colors(colors, proportions)

  # Display the colors used and the resulting color, along with the target color
  display_cmyk_colors(colors, result_color, target_color)