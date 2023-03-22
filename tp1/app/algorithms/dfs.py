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
  
    self.expanded_nodes += 1
    self.border_nodes -= 1
    
    # Visitar vecinos de la zona
    expansions = self.expand_zone(color)

    if expansions == 0:
      raise RuntimeError
    
    self.state.steps_to_state.append(color)

    # Imprimir estados intermedios
    # self.output_board(f"dfs", self.state.regions, color, cost)

    colors = self.state.regions[1].get_adjacent_colors(self.state)
    self.border_nodes += len(colors)

    # Si no hay adyacentes entonces es mi solucion
    if len(colors) == 0:
      self.solution_cost = cost + 1 # guardamos el costo de la solucion
      return True
    
    # Probamos con todos los colores hasta que alguno de ellos lleve a una solucion
    for c in colors:
      if self.search(c, cost+1):
        return True
    
    # Ningun color llevo a una solucion (no deberia pasar nunca)
    return False


  def solve(self):
    colors = self.state.regions[1].get_adjacent_colors(self.state)
    self.border_nodes += len(colors)
    # Probamos al inicio con todos los colores hasta que alguno de ellos lleve a una solucion
    for c in colors:
      if self.search(c, 0):
        break
    
    return self.state, self.solution_cost, self.expanded_nodes, self.border_nodes


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
