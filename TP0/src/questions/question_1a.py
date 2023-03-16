import matplotlib.pyplot as plt

from src.questions.helpers import build_dataframe_q1

def question_1_a(pokemon_factory, config):
    df = build_dataframe_q1(pokemon_factory, config)

    df = df.groupby("pokeball").agg({"accuracy": ["mean", "std"]}).reset_index()
    df.columns = df.columns.map("_".join)

    print(df)

    # Pokeball name to accuracy (mean dnd std)
    pokeballs = {}

    for pokeball in df["pokeball_"]:
        df_pokeball = df[df["pokeball_"] == pokeball]
        accuracy = df_pokeball["accuracy_mean"].values.tolist()[0]
        acc_std = df_pokeball["accuracy_std"].values.tolist()[0]
        pokeballs[pokeball] = {}
        pokeballs[pokeball]["mean"] = accuracy
        pokeballs[pokeball]["std"] = acc_std

    _, ax = plt.subplots()

    ax.bar(pokeballs.keys(), [pokeballs[pokeball]["mean"] for pokeball in pokeballs.keys()], yerr=[pokeballs[pokeball]["std"] for pokeball in pokeballs.keys()], align="center", alpha=0.5, ecolor="black", capsize=10)

    ax.set_ylabel("Accuracy")
    ax.grid()
    plt.savefig("./graphs/q_1_a.png")

    return df