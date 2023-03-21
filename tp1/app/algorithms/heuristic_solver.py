import math

from solver import Solver
from settings import settings

# Interfaz para los algoritmos de busqueda informados
class HeuristicSolver(Solver):
  def __init__(self, N, M):
    super().__init__(N, M)
    self.heuristic = settings.heuristic


  # Heuristica no admisible
  def prom_distinct_regions_heuristic(self, regions, colors):
    """Heuristica promedio de regiones distintas por color"""
    if colors == 0:
      return 0
    return math.floor(regions / colors)
  

  # TODO: agregar una heuristica admisible mas
  

  def distinct_colors_heuristic(self, colors):
    """Heuristica cantidad de colores distintos"""
    return colors


  def composite_heuristic(self, regions, colors):
    """Heuristica compuesta por las otras dos"""
    return max(
      self.prom_distinct_regions_heuristic(regions, colors),
      self.distinct_colors_heuristic(colors)
    )

  
  def get_heuristic(self, next_color):
    """
    Retorna la heuristica calculada para el estado que tendra el juego si se elige `next_color` en el proximo turno.
    """
    # Hacemos una copia para restaurar el estado del juego luego de calcular la heuristica
    _visited = self.visited.copy()
    _board = self.board.copy()
    _remaining_cells = self.remaining_cells

    self.visit_zone_neighbors(next_color)

    # Cantidad de regiones y colores distintos fuera de la zona
    regions, colors = self.get_number_of_regions()
    colors = len(colors)

    # Heuristica
    match self.heuristic:
      case "prom_distinct_regions":
        h = self.prom_distinct_regions_heuristic(regions, colors)
      case "distinct_colors":
        h = self.distinct_colors_heuristic(colors)
      case "composite":
        h = self.composite_heuristic(regions, colors)
    
    # Restauramos el estado del juego
    self.visited = _visited
    self.board = _board
    self.remaining_cells = _remaining_cells

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
  N = 4 # tamaño del tablero
  M = 2  # cantidad de colores

  heuristic_solver = HeuristicSolver(N, M)

  print("Tablero:")
  print(heuristic_solver.board)

  next_color = 1
  h = heuristic_solver.get_heuristic(next_color)

  regions, colors = heuristic_solver.get_number_of_regions()

  print(f"Regiones: {regions}, Colores: {len(colors)}")

  print(f"h(e) = {h}")
