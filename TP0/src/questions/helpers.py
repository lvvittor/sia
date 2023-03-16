from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect
from IPython.display import display
import pandas as pd
import numpy as np
import json

POKEBALLS = ["pokeball", "fastball", "ultraball", "heavyball"]

def build_dataframe_q1(pokemon_factory, config):
    attempts = config["question_1"]["catch_attempts"]
    df = pd.DataFrame(columns=["name", "pokeball", "accuracy", "error"])
    for pokemon in pokemon_factory.get_available_pokemons():
        new_pokemon = pokemon_factory.create(pokemon, config["question_1"]["pokemons_level"], StatusEffect.NONE, config["question_1"]["pokemons_health"]/100)
        for pokeball in POKEBALLS:
            accuracies = []
            for _ in range(100):
                catched = 0
                for _ in range(attempts):
                    attempt, rate = attempt_catch(new_pokemon, pokeball)
                    if attempt:
                        catched += 1
                accuracies.append(catched / attempts)
            df.loc[len(df)] = [pokemon, pokeball, np.mean(accuracies), np.std(accuracies)]
    return df

def build_dataframe_q2_a(pokemon_factory, config):
    attempts = config["question_2"]["catch_attempts"]
    pokemon = config["question_2"]["pokemon"]
    df = pd.DataFrame(columns=["name", "pokeball", "status", "accuracy", "error"])
    for status in StatusEffect.get_all():
        health = 1
        new_pokemon = pokemon_factory.create(pokemon, config["question_2"]["pokemons_level"], status, health)
        for pokeball in POKEBALLS:
            accuracies = []
            for _ in range(100):
                catched = 0
                for _ in range(attempts):
                    attempt, rate = attempt_catch(new_pokemon, pokeball)
                    if attempt:
                        catched += 1
                accuracies.append(catched / attempts)
            df.loc[len(df)] = [pokemon, pokeball, status.name, np.mean(accuracies), np.std(accuracies)]
    return df

def build_dataframe_q2_b(pokemon_factory, config):
    attempts = config["question_2"]["catch_attempts"]
    pokemon = config["question_2"]["pokemon"]
    df = pd.DataFrame(columns=["name", "pokeball", "health", "accuracy", "error"])
    for i in range(1,11):
        health = i/10
        new_pokemon = pokemon_factory.create(pokemon, config["question_2"]["pokemons_level"], StatusEffect.NONE, health)
        for pokeball in POKEBALLS:
            accuracies = []
            for _ in range(100):
                catched = 0
                for _ in range(attempts):
                    attempt, rate = attempt_catch(new_pokemon, pokeball)
                    if attempt:
                        catched += 1
                accuracies.append(catched / attempts)
            df.loc[len(df)] = [pokemon, pokeball, health, np.mean(accuracies), np.std(accuracies)]
    return df

def build_dataframe_q2_c(pokemon_factory, config):
    attempts = config["question_2"]["catch_attempts"]
    pokemon = config["question_2"]["pokemon"]
    df = pd.DataFrame(columns=["name", "pokeball", "level", "accuracy", "error"])
    for i in range(1,11):
        level = i*10
        health = 1
        new_pokemon = pokemon_factory.create(pokemon, level, StatusEffect.NONE, health)
        for pokeball in POKEBALLS:
            accuracies = []
            for _ in range(100):
                catched = 0
                for _ in range(attempts):
                    attempt, rate = attempt_catch(new_pokemon, pokeball)
                    if attempt:
                        catched += 1
                accuracies.append(catched / attempts)
            df.loc[len(df)] = [pokemon, pokeball, level, np.mean(accuracies), np.std(accuracies)]
    return df