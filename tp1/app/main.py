from services import BoardGeneratorService, BoardService, BenchMarkService
from settings import settings
from algorithms.dfs import solve_dfs
import pandas as pd

def solve_algorithm():
    match settings.algorithm:
        case "dfs":
            solve_dfs()

if __name__ == "__main__":
    board_generator = BoardGeneratorService(settings.board.N, settings.board.M)
    board = board_generator.generate()
    old_board = None

    i = 0
    while True:
        df = board_generator.dict_to_df(board.regions)
        print(df)

        # solve_algorithm()
        board_service = BoardService(df)

        with pd.option_context('display.max_rows', settings.board.N, 'display.max_columns', settings.board.N):
            print(f"Board:")
            print(board_service.get_board())
            board_service.set_colored_board("test"+str(i)+".png")
            i += 1
            print(f"Board size: {board_service.get_board_size()}")
            print(f"Board colors: {board_service.get_board_colors()}")
            print(f"Board color count: {board_service.get_board_color_count()}")

        new_color = input("What color do you want to change: ")
        if new_color == 'b':
            print("Backtracking")
            board = board_generator.undo_update()
            continue
        board = board_generator.update_state(new_color)
  


