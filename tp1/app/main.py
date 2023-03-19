from services import BoardGeneratorService, BoardService, BenchMarkService
from settings import settings
import dataframe_image as dfi
import pandas as pd
import numpy as np

if __name__ == "__main__":
    board_generator = BoardGeneratorService(settings.board.N, settings.board.M)
    board = board_generator.generate()
    board_service = BoardService(board)

    # df = pd.DataFrame(np.random.rand(6, 4))
    # df_styled = df.style.background_gradient()
    # print(df)
    # print(type(df_styled))
    # dfi.export(df_styled, f"{settings.Config.output_path}/test.png")

    with pd.option_context('display.max_rows', settings.board.N, 'display.max_columns', settings.board.N):
        print(f"Board:")
        print(board_service.get_board())

        # df_styled = board_service.get_colored_board()
        # print(type(df_styled))
        # dfi.export(df_styled, f"{settings.Config.output_path}/out.png")

        print(f"Board size: {board_service.get_board_size()}")
        print(f"Board colors: {board_service.get_board_colors()}")
        print(f"Board color count: {board_service.get_board_color_count()}")

