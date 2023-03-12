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

def answer_question(question, pokemon_factory):
    match question:
        case "1-a":
            question_1_a(pokemon_factory)
        case "1-b":
            pass

def question_1_a(pokemon_factory):
    for pokemon in pokemon_factory.get_available_pokemons():
        new_pokemon = pokemon_factory.create(pokemon, config["1_a"]["pokemons_level"], StatusEffect.NONE, config["1_a"]["pokemons_health"]/100)
        for pokeball in POKEBALLS:
            catched = 0
            for _ in range(100):
                attempt, rate = attempt_catch(new_pokemon, pokeball)
                if attempt:
                    catched += 1
