from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect
import pandas as pd

POKEBALLS = ["pokeball", "fastball", "ultraball", "heavyball"]

if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    config = json.load(open("config.json"))
    snorlax = factory.create("snorlax", 100, StatusEffect.NONE, 1)
    while True:
        question = input("What question should be answered?")
        answer_question(question, factory, config)
    """ print("No noise: ", attempt_catch(snorlax, "heavyball"))
    for _ in range(10):
        print("Noisy: ", attempt_catch(snorlax, "heavyball", 0.15)) """

def answer_question(question, pokemon_factory, config):
    match question:
        case "1-a":
            question_1_a(pokemon_factory, config)
        case "1-b":
            pass

def question_1_a(pokemon_factory, config):
    attempts = config["1_a"]["catch_attempts"]
    df = pd.Dataframe(columns=["name", "pokeball", "catched", "attempts"])
    for pokemon in pokemon_factory.get_available_pokemons():
        new_pokemon = pokemon_factory.create(pokemon, config["1_a"]["pokemons_level"], StatusEffect.NONE, config["1_a"]["pokemons_health"]/100)
        for pokeball in POKEBALLS:
            catched = 0
            for _ in range(attempts):
                attempt, rate = attempt_catch(new_pokemon, pokeball)
                if attempt:
                    catched += 1
        # Add row to dataframe
        df.loc[len(df)] = [pokemon, pokeball, catched, attempts]
    df["accuracy"] = df.catched / df.attempts
    df = df.groupby("pokeball").agg({"accuracy": ["mean", "std"]}).reset_index()
    df.columns = df.columns.map("_".join)
    display(df)

            
