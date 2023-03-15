from src.questions.helpers import build_dataframe_q1
from IPython.display import display

def question_1_a(pokemon_factory, config):
    df = build_dataframe_q1(pokemon_factory, config)
    df = df.groupby("pokeball").agg({"accuracy": ["mean", "std"]}).reset_index()
    df.columns = df.columns.map("_".join)
    display(df)
    return df