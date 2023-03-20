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


    while True:
        df = board_generator.dict_to_df(board)
        print(df)

        # solve_algorithm()
        board_service = BoardService(df)

        with pd.option_context('display.max_rows', settings.board.N, 'display.max_columns', settings.board.N):
            print(f"Board:")
            print(board_service.get_board())
            board_service.set_colored_board("test.png")
            print(f"Board size: {board_service.get_board_size()}")
            print(f"Board colors: {board_service.get_board_colors()}")
            print(f"Board color count: {board_service.get_board_color_count()}")

        new_color = input("What color do you want to change: ")
        board = board_generator.update_state(new_color)
    


