from solver import Solver
from services.board_generator_service import BoardGeneratorService

class BFS(Solver):
    def __init__(self, N, M):
        super().__init__(N, M)
    
    def search(self, board):
        """
        Busca una solucion iterativamente.
        Devuelve True si la encontro, False en caso contrario.
        """
        
        cost = 0

        # put numbers from 0 to M in queue 
        colors = [i for i in range(1, self.M+1)]
        queue = [colors]

        # amount of regions
        regions = len(board.state.regions)
        
        while queue:
            # get first color
            color = queue.pop(0)
            cost += 1

            # try to update board state with that color
            board.update_state(color)

            # if no new regions were merged, discard this path
            if len(board.state.regions) == regions:
                board.undo_update()
                cost -= 1
                continue

            # some regions were merged, check if we found a solution
            if len(board.state.regions) == 1:
                return True, cost
            
            # if not, add new colors to queue
            map(queue.append, colors)
        
        # no solution found
        return False  

    def solve(self):
        """
        Genera un tablero inicial y lo resuelve.
        """
        board = BoardGeneratorService(self.N, self.M).generate()
        return self.search(board)

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