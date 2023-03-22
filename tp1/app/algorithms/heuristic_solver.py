from queue import PriorityQueue
from region import State
from solver import Solver
from settings import settings

# Interfaz para los algoritmos de busqueda informados
class HeuristicSolver(Solver):
  def __init__(self, N, M):
    super().__init__(N, M)
    self.heuristic = settings.heuristic

  
  def distinct_colors_heuristic(self, state: State):
    amount_colors = 0
    colors_visited = []
    for region in filter(lambda region: region.id != 1, state.values()):
      if region.color not in colors_visited:
        colors_visited.append(region.color)
        amount_colors += 1
      if amount_colors == settings.board.M:
        break

    return amount_colors
  

  def dijkstra(state: State):
    zone_id = 1
    distances = {region_id:float('inf') for region_id in state.regions.keys()}

    distances[zone_id] = 0
    visited_ids = []
    pq = PriorityQueue()
    pq.put((0, zone_id))

    while not pq.empty():
        _, current_region_id = pq.get()
        visited_ids.append(current_region_id)

        for adjacent in state.regions.keys():
            if adjacent in state.regions[current_region_id].adjacents:
                distance = 1 # El costo de pasar de una region a otra es siempre 1 en nuestro caso
                if adjacent not in visited_ids:
                    old_cost = distances[adjacent]
                    new_cost = distances[current_region_id] + distance
                    if new_cost < old_cost:
                        pq.put((new_cost, adjacent))
                        distances[adjacent] = new_cost
    return distances
  

  def get_maximum_distance(distances):
    return max(distances.values())


  def farthest_region_heuristic(self, state: State):
    return self.get_maximum_distance(self.dijkstra(state))


  def composite_heuristic(self, state: State):
    """Heuristica compuesta por las otras dos"""
    return max(
      self.farthest_region_heuristic(state),
      self.distinct_colors_heuristic(state)
    )


  def get_heuristic(self, state: State):
    """
    Retorna la heuristica calculada para un estado.
    """

    match self.heuristic:
      case "farthest_region_heuristic":
        h = self.farthest_region_heuristic(state)
      case "distinct_colors":
        h = self.distinct_colors_heuristic(state)
      case "composite":
        h = self.composite_heuristic(state)

    return h

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
