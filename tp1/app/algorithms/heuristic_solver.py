from solver import Solver

# Interfaz para los algoritmos de busqueda informados
class HeuristicSolver(Solver):
  def __init__(self, N, M):
    super().__init__(N, M)

  
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
