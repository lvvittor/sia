from algorithms.solver import Solver
from settings import settings
from services.board_generator_service import BoardGeneratorService
from services.board_service import BoardService

class BFS(Solver):
    FIRST_COLOR = 0


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
        color_queue = [c for c in colors]

        # save pending states while expanding a level of the tree
        state_queue = [[self.state.copy(), self.solution_cost]]

        # output initial state
        # self.output_board(f"0-bfs", self.state.regions, self.initial_color, self.solution_cost)

        file_prefix = 1
        
        while color_queue:
            # get first color
            color = color_queue.pop(0)

            # if we are checking the first color, then we are checking a new state (i.e. a new level of the tree for a given node/color)
            # we always start from the first color, even if the parent node expanded that same color
            if color == BFS.FIRST_COLOR:
                self.state, self.solution_cost = state_queue.pop(0)
                    

            state_copy = self.state.copy()
            # try to update board state with that color
            expansions = self.expand_zone(color)

            # some regions were merged, check if we found a solution
            if self.is_solution():
                # output final state
                self.solution_cost += 1
                # self.output_board(f"{file_prefix}-bfs", self.state.regions, color, self.solution_cost)
                return True
            elif expansions == 0: # if no new regions were merged, discard this path
                self.state = state_copy
                continue
            else:
                # an expansion has been made, save the state
                state_queue.append([self.state.copy(), self.solution_cost+1])

                file_prefix += 1

                # add all the colors for this expanded state to the queue (we'll check them all when we come back to this state later)
                for c in colors:
                    color_queue.append(c)

                # output intermediate states
                # self.output_board(f"{file_prefix}-bfs", self.state.regions, color, self.solution_cost+1)

                # rollback to the parent state to expand the rest of the colors
                self.state = state_copy

                
        
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