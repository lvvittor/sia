from src.questions.helpers import build_dataframe_q2_c
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

colors = {
    "pokeball": 'r',
    "fastball": 'g',
    "ultraball": 'b',
    "heavyball": 'y',
}

def question_2_c(pokemon_factory, config):
    df = build_dataframe_q2_c(pokemon_factory, config)
    display(df)

    for pokeball in df["pokeball"].unique():
        df_pokeball = df[df["pokeball"] == pokeball]
        plt.plot(df_pokeball['level'], df_pokeball['accuracy'], color=colors[pokeball], marker='o', label=pokeball)
        plt.errorbar(df_pokeball['level'], df_pokeball['accuracy'], df_pokeball['error'], fmt='none', color=colors[pokeball], capsize=3)

    plt.title('accuracy vs level for '+ config["question_2"]["pokemon"], fontsize=14)
    plt.xlabel('level', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.grid(True)
    plt.savefig("./graphs/q_2_c.png", bbox_inches='tight')
    plt.close()
    return df