from src.questions.question_1a import question_1_a
from src.questions.helpers import build_dataframe_q1
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt

def question_1_b(pokemon_factory, config):
    df = build_dataframe_q1(pokemon_factory, config)

    pokeballs = {}

    # get accuracy for each pokeball
    for pokeball in df["pokeball"].unique():
        df_pokeball = df[df["pokeball"] == pokeball]
        pokeballs[pokeball] = df_pokeball["accuracy"].values.tolist()
    
    pokemons = pokemon_factory.get_available_pokemons()

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
    
    # Set position of bar on X axis
    br1 = np.arange(len(pokeballs["pokeball"]))
    br1 = [x + 0.1*x for x in br1] # add some space between pokemons
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    
    # Make the plot
    plt.bar(br1, pokeballs["pokeball"], color ='r', width = barWidth,
            edgecolor ='grey', label ='Pokeball')
    plt.bar(br2, pokeballs["fastball"], color ='g', width = barWidth,
            edgecolor ='grey', label ='Fastball')
    plt.bar(br3, pokeballs["ultraball"], color ='b', width = barWidth,
            edgecolor ='grey', label ='Ultraball')
    plt.bar(br4, pokeballs["heavyball"], color ='y', width = barWidth,
            edgecolor ='grey', label ='Heavyball')
    
    # Adding Xticks
    plt.xlabel('Pokemons', fontweight ='bold', fontsize = 15)
    plt.ylabel('Pokeballs\' catching probability', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth*1.5 + 0.1*r for r in range(len(pokeballs["pokeball"]))], pokemons) # put the name in between the bars
    
    plt.legend()
    plt.savefig("./graphs/q_1_b.png")
