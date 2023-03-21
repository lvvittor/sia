from services import BoardGeneratorService, BoardService, BenchMarkService
from settings import settings
from algorithms import DFS
import pandas as pd

def solve_algorithm():
    match settings.algorithm:
        case "dfs":
            dfs_solver = DFS(settings.board.N, settings.board.M)
            board, cost = dfs_solver.solve()

if __name__ == "__main__":
    board_generator = BoardGeneratorService(settings.board.N, settings.board.M)
    board = board_generator.generate()

    i = 0
    while True:
        df = board_generator.dict_to_df(board.regions)

        # solve_algorithm()
        board_service = BoardService(df)

        with pd.option_context('display.max_rows', settings.board.N, 'display.max_columns', settings.board.N):
            print(board_service.get_board())
        
        print(f"Board [colors={board_service.get_board_colors()}, len(colors)={board_service.get_board_color_count()}]")
        
        board_service.set_colored_board(f"test{i}.png")
        i += 1

        new_color = input("What color do you want to change: ")
        if new_color == "b":
            print("Backtracking")
            board = board_generator.undo_update()
            continue

        board = board_generator.update_state(int(new_color))
  


