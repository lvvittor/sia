from src.questions.question_1a import question_1_a
from src.questions.helpers import build_dataframe_q1
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt

def question_1_b(pokemon_factory, config):
    df = build_dataframe_q1(pokemon_factory, config)
    
    pokemons = pokemon_factory.get_available_pokemons()

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
    
    # set height of bar
    POKEBALL_PROBS = [12, 30, 1, 8, 22]
    FASTBALL_PROBS = [28, 6, 16, 5, 10]
    ULTRABALL_PROBS = [29, 3, 24, 25, 17]
    HEAVYBALL_PROBS = [29, 3, 24, 25, 17]
    
    # Set position of bar on X axis
    br1 = np.arange(len(POKEBALL_PROBS))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    
    # Make the plot
    plt.bar(br1, POKEBALL_PROBS, color ='r', width = barWidth,
            edgecolor ='grey', label ='Pokeball')
    plt.bar(br2, FASTBALL_PROBS, color ='g', width = barWidth,
            edgecolor ='grey', label ='Fastball')
    plt.bar(br3, ULTRABALL_PROBS, color ='b', width = barWidth,
            edgecolor ='grey', label ='Ultraball')
    plt.bar(br4, HEAVYBALL_PROBS, color ='y', width = barWidth,
            edgecolor ='grey', label ='Heavyball')
    
    # Adding Xticks
    plt.xlabel('Pokemons', fontweight ='bold', fontsize = 15)
    plt.ylabel('Pokeballs\' catching probability', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(POKEBALL_PROBS))], pokemons)
    
    plt.legend()
    plt.savefig("./graphs/q_1_b.png")
