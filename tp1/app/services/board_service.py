import pandas as pd
import seaborn as sns
import numpy as np
import dataframe_image as dfi
from settings import settings

class BoardService():
    def __init__(self, board: pd.DataFrame):
        self.board = board

    def get_board(self):
        return self.board
    
    def get_board_size(self):
        return self.board.shape[0]
    
    def get_board_colors(self):
        return np.unique(self.board.values)
    
    def get_board_color_count(self):
        return len(self.get_board_colors())
    
    def get_distinct_colors(self):
        return self.board.nunique().sum()
    
    def set_colored_board(self, filename: str):
        df = self.board

        # get unique values
        codes, unique = pd.factorize(df.stack())

        # generate a color palette with as many colors as there are unique values
        palette = sns.color_palette(None, len(unique)).as_hex()

        # map the unique values to the colors
        reshape_df = pd.DataFrame(
                        codes.reshape(df.shape),
                        index=df.index, 
                        columns=df.columns
        ).replace(dict(enumerate(palette))).radd('background-color: ')

        df.loc[:] = ''

        # apply the colors to the cells
        def get_colors(item):
            return reshape_df

        df_styled = df.style.apply(get_colors, axis=None)

        dfi.export(df_styled, f"{settings.Config.output_path}/{filename}")

        
         
        