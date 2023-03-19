import pandas as pd
import seaborn as sns
import numpy as np

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
    
    def get_colored_board(self):
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
    
        return df.style.apply(reshape_df, axis=None)

        
         
        