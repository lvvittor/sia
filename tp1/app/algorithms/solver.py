from services.board_service import BoardService
import numpy as np

from services.board_generator_service import BoardGeneratorService
from settings import settings

# Interfaz para los algoritmos de busqueda no informados
class Solver:
  def __init__(self, state):
    # Estado del juego
    self.state = state
    self.solution_cost = 0
    self.board_service = BoardService()
    # Inicializamos la zona
    self.initial_color = self.state.regions[1].color

  def is_solution(self):
    """
    Devuelve True si el estado actual es una solucion, False en caso contrario.
    """
    return len(self.state.regions) == 1 

  def expand_zone(self, color):
    """
    Visita todas las regiones vecinas de la zona, dado un cambio de color.
    Devuelve la cantidad de vecinos agregados a la zona.
    """
    _, expansions = self.state.update_state(color)
    
    return expansions


  def search(self, color, cost):
    """
    Busca una solucion al problema, con cierto metodo de busqueda.
    Devuelve True si la encontro, False en caso contrario.
    Actualiza `self.solution_cost` con el costo de la solucion encontrada.
    """
    raise NotImplementedError


  def solve(self):
    """
    Hace un setup inicial y busca una solucion al problema.
    Retorna el tablero solucion y el costo de la solucion.
    """
    raise NotImplementedError
  
  def output_board(self, algorithm_name: str, regions: dict, color: int, cost: int):
    # TODO: return when `visualization` config is False
    board_generator = BoardGeneratorService(settings.board.N, settings.board.M)
    state_df = board_generator.dict_to_df(regions)
    print()
    print(state_df)
    self.board_service.set_colored_board(state_df, f"{algorithm_name}[cost:{cost},color:{color}].png")
    print()
