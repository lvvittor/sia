from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any
from algorithms.heuristic_solver import HeuristicSolver
from settings import settings

# esta clase hace falta porque el state no es comparable.
@dataclass(order=True)
class PrioritizedItem:
  priority: int
  item: Any=field(compare=False)

class Greedy(HeuristicSolver):
  def __init__(self, state):
    super().__init__(state)

  def search(self):
    colors = [i for i in range(0, settings.board.M)]
    pq = PriorityQueue()

    # Le pongo valor > 0 para que no frene en la primer iteracion, el valor de la heuristica de la root no afecta en nada.
    pq.put(PrioritizedItem(1, self.state))

    while not pq.empty():
      # pq.get() son tuplas donde [0] es el valor de la heuristica, [1] es el estado.
      next_item = pq.get()
      next_state = next_item.item
      
      # h(e) = 0 termina el algirtmo y encontro solucion
      if next_item.priority == 0:
        self.state = next_state
        self.solution_cost = next_state.cost
        return True


      # por cada color agrego un nuevo nodo a la frontera
      for color in colors:
        # hago una copia de cada estado para guardar distintos estados en la queue
        new_state = next_state.copy()
        _, expansions = new_state.update_state(color)

        # tengo que guardar el costo en el estado al tener muchos estados distintos que pueden tener distintos costos
        # aumento de a 1 porque cada una de nuestras aristas vale 1
        new_state.increase_cost(1)

        # si el estado cambio entonces lo agrego a la cola
        if expansions > 0:
          pq.put(PrioritizedItem(self.get_heuristic(new_state), new_state))

    # no encontro solucion
    return False


  def solve(self):
    if self.search():
        return self.state, self.solution_cost
    return None, None