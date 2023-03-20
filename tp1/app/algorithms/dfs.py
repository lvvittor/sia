import numpy as np

N = 20 # tamaño del tablero
M = 5  # cantidad de colores

# Ver ejemplo de uso al final del archivo

class DFS():
  # Generico para todos los metodos de busqueda
  def __init__(self, N, M):
    # Matriz de colores
    self.board = np.random.randint(1, M+1, (N, N))
    # Matriz de visitados == celdas pertenecientes a la zona (0: no visitado, 1: visitado)
    self.visited = np.zeros((N, N))
    self.remaining_cells = N*N
    self.solution_cost = 0

    # Inicializamos la zona
    self.initial_color = self.board[0][0]
    self.visit_neighbors(0, 0, self.initial_color)


  # Generico para todos los metodos de busqueda
  def visit_neighbors(self, i, j, color):
    """
    Visita los vecinos de la casilla (i, j) que sean del mismo color.
    Devuelve la cantidad de vecinos nuevos visitados.
    """
    if i < 0 or i >= N or j < 0 or j >= N:
      return 0
    if self.visited[i][j] == 1:
      return 0
    if self.board[i][j] != color:
      return 0
    
    self.visited[i][j] = 1
    self.remaining_cells -= 1

    new_visited = 1

    new_visited += self.visit_neighbors(i-1, j, color)
    new_visited += self.visit_neighbors(i+1, j, color)
    new_visited += self.visit_neighbors(i, j-1, color)
    new_visited += self.visit_neighbors(i, j+1, color)

    return new_visited


  # Especifico de cada metodo de busqueda
  def search(self, color, cost):
    """
    Busca una solucion recursivamente.
    Devuelve True si la encontro, False en caso contrario.
    """
    if self.remaining_cells == 0:
      self.solution_cost = cost # guardamos el costo de la solucion
      return True
    
    # Visitar vecinos de la zona
    visited = 0
    for i in range(N):
      for j in range(N):
        if self.visited[i][j] == 1:
          visited += self.visit_neighbors(i, j, color)
    
    # Si no se visitaron nuevas celdas, descartamos este camino
    if visited == 0:
      return False
    
    # Probamos con todos los colores hasta que alguno de ellos lleve a una solucion
    for c in range(1, M+1):
      if self.search(c, cost+1):
        return True
    
    # Ningun color llevo a una solucion (no deberia pasar nunca)
    return False


# Ejemplo de busqueda DFS
dfs_solver = DFS(N, M)
dfs_solver.search(dfs_solver.initial_color, 0)
print(f"Costo de la solucion: {dfs_solver.solution_cost}")
