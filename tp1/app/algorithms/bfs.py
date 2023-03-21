from algorithms.solver import Solver
from settings import settings
from services.board_generator_service import BoardGeneratorService
from services.board_service import BoardService

class BFS(Solver):
    def __init__(self, state):
        super().__init__(state)
    
    def search(self):
        """
        Busca una solucion iterativamente.
        Devuelve True si la encontro, False en caso contrario.
        """

        # all colors
        colors = [i for i in range(0, settings.board.M)]

        # append colors to color_queue
        color_queue = []
        for color in colors:
            color_queue.append(color)

        state_queue = [self.state]
        current_state = None
        
        while color_queue:
            # get first color
            color = color_queue.pop(0)

            # if we are checking the first color, then we are checking a new state
            if color == 0:
                current_state = state_queue.pop(0)
                self.solution_cost += 1

            # try to update board state with that color
            expansions = self.expand_zone(color)

            # if no new regions were merged, discard this path
            if expansions == 0:
                self.state = current_state
                continue

            # some regions were merged, check if we found a solution
            if len(self.state.regions) == 1:
                return True
            
            # an expansion has been made, save the state and add colors to queue
            state_queue.append(self.state)
            for color in colors:
                color_queue.append(color)

            # print intermediate states
            board_generator = BoardGeneratorService(settings.board.N, settings.board.M)
            state_df = board_generator.dict_to_df(self.state.regions)
            print()
            print(state_df)
            self.board_service.set_colored_board(state_df, f"bfs{self.solution_cost}-{color}.png")
            print()

            # rollback for next color check
            self.state = current_state
        
        # no solution found
        return False

    def solve(self):
        """
        Genera un tablero inicial y lo resuelve.
        """
        if self.search():
            return self.state, self.solution_cost
        return None, None

# Ejemplo de uso (tamaño original: 14x14, 6 colores)
if __name__ == "__main__":
    N = 14 # tamaño del tablero
    M = 6  # cantidad de colores

    bfs_solver = BFS(N, M)

    board = BoardGeneratorService(N, M).generate()
    print("Tablero inicial:")
    
    solution, cost = bfs_solver.search(board)

    if solution:
        print(f"Costo de la solucion: {cost}")
    else:
        print("No se encontro solucion")