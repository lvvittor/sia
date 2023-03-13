from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect
from IPython.display import display
import pandas as pd
import json

POKEBALLS = ["pokeball", "fastball", "ultraball", "heavyball"]

def build_dataframe_q1(pokemon_factory, config):
    attempts = config["question_1"]["catch_attempts"]
    df = pd.DataFrame(columns=["name", "pokeball", "catched", "attempts"])
    for pokemon in pokemon_factory.get_available_pokemons():
        new_pokemon = pokemon_factory.create(pokemon, config["question_1"]["pokemons_level"], StatusEffect.NONE, config["question_1"]["pokemons_health"]/100)
        for pokeball in POKEBALLS:
            catched = 0
            for _ in range(attempts):
                attempt, rate = attempt_catch(new_pokemon, pokeball)
                if attempt:
                    catched += 1
            df.loc[len(df)] = [pokemon, pokeball, catched, attempts]
    df["accuracy"] = df.catched / df.attempts
    return df