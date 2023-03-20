import json

from solver import Solver

# Interfaz para los algoritmos de busqueda informados
class HeuristicSolver(Solver):
  def __init__(self, N, M):
    super().__init__(N, M)
    config = json.load(open("tp1/config.json")) # TODO: use settings instead of this hardcoded path
    self.heuristic = config["heuristic"]

  
  def get_heuristic(self):
    """
    Retorna la heuristica para el estado actual del juego.
    """
    # Cantidad de regiones y colores distintos fuera de la zona
    regions, colors = self.get_number_of_regions()
    colors = len(colors)

    # Heuristica
    match self.heuristic:
      case "prom_distinct_regions":
        h = regions / colors
      case "distinct_colors":
        h = colors
      case "composite":
        h = max(regions / colors, colors)

    return h


  def get_number_of_regions(self):
    """
    Retorna la cantidad de regiones fuera de la zona y los colores que las componen.
    """
    # Hacemos una copia para restaurar el estado del juego luego de calcular la heuristica
    _visited = self.visited.copy()

    regions = 0
    colors = set()

    for i in range(self.N):
      for j in range(self.N):
        if self.visited[i][j] == 0:
          regions += 1
          self.consume_region(i, j, self.board[i][j])
          colors.add(self.board[i][j])

    # Restauramos el estado del juego
    self.visited = _visited

    return regions, colors


  def consume_region(self, i, j, color):
    """Visita toda una region"""
    if i < 0 or i >= self.N or j < 0 or j >= self.N:
      return
    if self.visited[i][j] == 1:
      return
    if self.board[i][j] != color:
      return

    self.visited[i][j] = 1

    self.consume_region(i-1, j, color)
    self.consume_region(i+1, j, color)
    self.consume_region(i, j-1, color)
    self.consume_region(i, j+1, color)


# Test
if __name__ == "__main__":
  N = 14 # tamaño del tablero
  M = 6  # cantidad de colores

  heuristic_solver = HeuristicSolver(N, M)

  h = heuristic_solver.get_heuristic()

  print(f"h(e) = {h}")
