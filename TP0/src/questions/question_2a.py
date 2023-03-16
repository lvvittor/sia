from src.questions.helpers import build_dataframe_q2_a
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

colors = {
    "pokeball": 'r',
    "fastball": 'g',
    "ultraball": 'b',
    "heavyball": 'y',
}

def question_2_a(pokemon_factory, config):
    df = build_dataframe_q2_a(pokemon_factory, config)
    display(df)

    for pokeball in df["pokeball"].unique():
        df_pokeball = df[df["pokeball"] == pokeball]
        plt.plot(df_pokeball['status'], df_pokeball['accuracy'], color=colors[pokeball], marker='o', label=pokeball)
        plt.errorbar(df_pokeball['status'], df_pokeball['accuracy'], df_pokeball['error'], fmt='none', color=colors[pokeball], capsize=3)

    plt.title('accuracy vs status for '+ config["question_2"]["pokemon"], fontsize=14)
    plt.xlabel('status', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig("./graphs/q_2_a.png")
    plt.close()
    return df