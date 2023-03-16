from src.questions.helpers import build_dataframe_q2_b
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

colors = {
    "pokeball": 'r',
    "fastball": 'g',
    "ultraball": 'b',
    "heavyball": 'y',
}

def question_2_b(pokemon_factory, config):
    df = build_dataframe_q2_b(pokemon_factory, config)
    display(df)

    for pokeball in df["pokeball"].unique():
        df_pokeball = df[df["pokeball"] == pokeball]
        plt.plot(df_pokeball['health'], df_pokeball['accuracy'], color=colors[pokeball], marker='o', label=pokeball)
        plt.errorbar(df_pokeball['health'], df_pokeball['accuracy'], df_pokeball['error'], fmt='none', color=colors[pokeball], capsize=3)

    plt.title('accuracy vs health for '+ config["question_2"]["pokemon"], fontsize=14)
    plt.xlabel('health', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig("./graphs/q_2_b.png")
    plt.close()
    return df