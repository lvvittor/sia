from algorithms.heuristic_solver import HeuristicSolver

class Greedy(HeuristicSolver):
  def __init__(self, state, heuristic=settings.heuristic):
    super().__init__(state, heuristic)


  def add_to_priority_queue(self, pq, state):
    pq.put(self.PrioritizedItem(self.get_heuristic(state), state))


  def found_solution(self, priority):
    return priority == 0

  def solve(self):
    if self.search():
        return self.state, self.solution_cost, self.expanded_nodes, self.border_nodes
    return None, None, 0, 0