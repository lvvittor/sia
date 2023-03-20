import numpy as np

# Define an interface for a game solver
class Solver:
  def __init__(self, N, M):
    self.N = N
    self.M = M
    # Matriz de colores
    self.board = np.random.randint(1, M+1, (N, N))
    # Matriz de visitados == celdas pertenecientes a la zona (0: no visitado, 1: visitado)
    self.visited = np.zeros((N, N))
    self.remaining_cells = N*N
    self.solution_cost = 0

    # Inicializamos la zona
    self.initial_color = self.board[0][0]
    self.visit_neighbors(0, 0, self.initial_color)


  def visit_neighbors(self, i, j, color):
    """
    Visita los vecinos de la casilla (i, j) que sean del mismo color.
    Devuelve la cantidad de vecinos nuevos visitados.
    """
    if i < 0 or i >= self.N or j < 0 or j >= self.N:
      # print(f"Fuera de rango: ({i}, {j})")
      return 0
    
    # Indica si la celda ya esta en la zona pero con otro color, en cuyo caso le actualizamos el color
    already_visited = False

    if self.visited[i][j] == 1:
      if self.board[i][j] != color:
        self.board[i][j] = color
        already_visited = True
      else:
        # print(f"Ya visitado: ({i}, {j})")
        return 0

    if self.board[i][j] != color:
      # print(f"Distinto color: ({i}, {j}) {self.board[i][j]} != {color}")
      return 0
    
    # print(f"Visitando: ({i}, {j})")

    if not already_visited:
      self.visited[i][j] = 1
      self.remaining_cells -= 1

    if already_visited:
      new_visited = 0
    else:
      new_visited = 1

    new_visited += self.visit_neighbors(i-1, j, color)
    new_visited += self.visit_neighbors(i+1, j, color)
    new_visited += self.visit_neighbors(i, j-1, color)
    new_visited += self.visit_neighbors(i, j+1, color)

    return new_visited


  def visit_zone_neighbors(self, color):
    """
    Visita todos los vecinos de la zona, dado un cambio de color.
    Devuelve la cantidad de vecinos agregados a la zona.
    """
    visited = 0
    for i in range(self.N):
      for j in range(self.N):
        if self.visited[i][j] == 1:
          visited += self.visit_neighbors(i, j, color)
    
    return visited


  def search(self, color, cost):
    """
    Busca una solucion al problema, con cierto metodo de busqueda.
    Devuelve True si la encontro, False en caso contrario.
    Actualiza `self.solution_cost` con el costo de la solucion encontrada.
    """
    raise NotImplementedError


  def solve(self):
    """
    Hace un setup inicial y busca una solucion al problema.
    Retorna el tablero solucion y el costo de la solucion.
    """
    raise NotImplementedError
