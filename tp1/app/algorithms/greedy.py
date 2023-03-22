from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any
from algorithms.heuristic_solver import HeuristicSolver
from settings import settings


class Greedy(HeuristicSolver):
  def __init__(self, state, heuristic=settings.heuristic):
    super().__init__(state, heuristic)


  def add_to_priority_queue(self, pq, state):
    pq.put(self.PrioritizedItem(self.get_heuristic(state), state))


  def found_solution(self, priority):
    return priority == 0

  def solve(self):
    if self.search():
        return self.state, self.solution_cost
    return None, None