import numpy as np

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


  # Especifico de cada metodo de busqueda
  def search(self, color, cost):
    """
    Busca una solucion recursivamente.
    Devuelve True si la encontro, False en caso contrario.
    """
    if self.remaining_cells == 0:
      self.solution_cost = cost # guardamos el costo de la solucion
      return True
    
    # Guardamos el estado actual del tablero en caso de que necesitemos rollbackear (i.e. actualizamos el color de toda la zona pero en realidad no habia vecinos nuevos)
    board_copy = self.board.copy()
    
    # Visitar vecinos de la zona
    visited = 0
    for i in range(N):
      for j in range(N):
        if self.visited[i][j] == 1:
          visited += self.visit_neighbors(i, j, color)
    
    # Si no se visitaron nuevas celdas, descartamos este camino
    if visited == 0:
      # print(f"Pincho con color {color}")
      self.board = board_copy # rollback
      return False
    
    # Probamos con todos los colores hasta que alguno de ellos lleve a una solucion
    for c in range(1, M+1):
      if c != color and self.search(c, cost+1):
        return True
    
    # Ningun color llevo a una solucion (no deberia pasar nunca)
    return False


# Ejemplo de uso (tamaño original: 14x14, 6 colores)

N = 14 # tamaño del tablero
M = 6  # cantidad de colores

dfs_solver = DFS(N, M)

print("Tablero inicial:")
print(dfs_solver.board)

# Probamos al inicio con todos los colores hasta que alguno de ellos lleve a una solucion
for c in range(1, M+1):
  # print(f"Probando con color {c}")
  if dfs_solver.search(c, 0) == True:
    break

print("Tablero solucion:")
print(dfs_solver.board)

print(f"Costo de la solucion: {dfs_solver.solution_cost}")
