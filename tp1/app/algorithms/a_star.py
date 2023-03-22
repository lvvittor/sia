from algorithms.heuristic_solver import HeuristicSolver

class AStar(HeuristicSolver):
  def __init__(self, state):
    super().__init__(state)

  def add_to_priority_queue(self, pq, state):
    h = self.get_heuristic(state)
    pq.put(self.PrioritizedItem((h+state.cost, h), state))


  def found_solution(self, priority):
    return priority[1] == 0


  def solve(self):
    if self.search():
        return self.state, self.solution_cost
    return None, None