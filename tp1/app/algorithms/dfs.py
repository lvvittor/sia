from solver import Solver

# Ver ejemplo de uso al final del archivo

class DFS(Solver):
  def __init__(self, N, M):
    super().__init__(N, M)


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
    visited = self.visit_zone_neighbors(color)
    
    # Si no se visitaron nuevas celdas, descartamos este camino
    if visited == 0:
      # print(f"Pincho con color {color}")
      self.board = board_copy # rollback
      return False
    
    # Probamos con todos los colores hasta que alguno de ellos lleve a una solucion
    for c in range(1, self.M+1):
      if c != color and self.search(c, cost+1):
        return True
    
    # Ningun color llevo a una solucion (no deberia pasar nunca)
    return False


  def solve(self):
    # Probamos al inicio con todos los colores hasta que alguno de ellos lleve a una solucion
    for c in range(1, self.M+1):
      # print(f"Probando con color {c}")
      if self.search(c, 0):
        break
    
    return self.board, self.solution_cost


# Ejemplo de uso (tamaño original: 14x14, 6 colores)
if __name__ == "__main__":
  N = 14 # tamaño del tablero
  M = 6  # cantidad de colores

  dfs_solver = DFS(N, M)

  print("Tablero inicial:")
  print(dfs_solver.board)

  solution, cost = dfs_solver.solve()

  print("Tablero solucion:")
  print(solution)

  print(f"Costo de la solucion: {cost}")
