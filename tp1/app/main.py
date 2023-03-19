from services import BoardGeneratorService
from settings import settings
from algorithms.dfs import solve_dfs
from region import RegionFactory

def solve_algorithm():
    match settings.algorithm:
        case "dfs":
            solve_dfs()


if __name__ == "__main__":
    board_generator = BoardGeneratorService(settings.board.N, settings.board.M)
    board = board_generator.generate()
    regions = RegionFactory.create(board)
    print(board)
    solve_algorithm()
