import pandas as pd
from services.board_generator_service import BoardGeneratorService
from region import State
import seaborn as sns
import numpy as np
import dataframe_image as dfi
from settings import settings

class BoardService:
    def __init__(self):
        palette = sns.color_palette(None, settings.board.M).as_hex()
        self.dict_pallete = dict(enumerate(palette))
    
    def set_colored_board(self, df: pd.DataFrame, filename: str):

        if settings.visualization == False:
            return

        # map the unique values to the colors
        reshape_df = pd.DataFrame(
                        df,
                        index=df.index, 
                        columns=df.columns
        ).replace(self.dict_pallete).radd('background-color: ')
        df.loc[:] = ''

        # apply the colors to the cells
        def get_colors(item):
            return reshape_df

        df_styled = df.style.apply(get_colors, axis=None)

        dfi.export(df_styled, f"{settings.Config.output_path}/{filename}")

    def print_solution(self, board_generator: BoardGeneratorService, initial_state: State, steps: list[int]):
        state = initial_state.copy()
        i = 0
        for step in steps:
            df = board_generator.dict_to_df(state.regions)
            self.set_colored_board(df, f"solution{i}.png")
            i += 1
            state.update_state(step)
        df = board_generator.dict_to_df(state.regions)
        self.set_colored_board(df, f"solution{i}.png")
         
        