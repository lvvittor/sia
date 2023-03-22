from algorithms.solver import Solver
from settings import settings
from services.board_generator_service import BoardGeneratorService
from services.board_service import BoardService

# Ver ejemplo de uso al final del archivo

class DFS(Solver):
  def __init__(self, state):
    super().__init__(state)


  # Especifico de cada metodo de busqueda
  def search(self, color, cost):
    """
    Busca una solucion recursivamente.
    Devuelve True si la encontro, False en caso contrario.
    """
    if len(self.state.regions) == 1:
      self.solution_cost = cost # guardamos el costo de la solucion
      return True
    
    # Guardamos el estado actual del tablero en caso de que necesitemos rollbackear (i.e. actualizamos el color de toda la zona pero en realidad no habia vecinos nuevos)
    state_copy = self.state.copy()
    
    # Visitar vecinos de la zona
    expansions = self.expand_zone(color)
    
    # Si no se visitaron nuevas celdas, descartamos este camino
    if expansions == 0:
      self.state = state_copy # rollback
      return False
    
    # Imprimir estados intermedios
    self.output_board(f"dfs", self.state.regions, color, cost)
    
    # Probamos con todos los colores hasta que alguno de ellos lleve a una solucion
    for c in range(0, settings.board.M):
      if c != color and self.search(c, cost+1):
        return True
    
    # Ningun color llevo a una solucion (no deberia pasar nunca)
    return False


  def solve(self):
    # Probamos al inicio con todos los colores hasta que alguno de ellos lleve a una solucion
    for c in range(0, settings.board.M):
      if self.search(c, 0):
        break
    
    return self.state, self.solution_cost


# Ejemplo de uso (tamaño original: 14x14, 6 colores)
if __name__ == "__main__":
  board_generator = BoardGeneratorService(settings.board.N, settings.board.M)
  initial_state = board_generator.generate()
  board_service = BoardService()

  dfs_solver = DFS(state)

  print("Tablero inicial:")
  initial_df = board_generator.dict_to_df(initial_state.regions)
  print(initial_df)

  solution, cost = dfs_solver.solve()

  solution_df = board_generator.dict_to_df(solution.regions)
  print("Tablero solucion:")
  print(solution_df)

  print(f"Costo de la solucion: {cost}")
