from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect
from src.questions.question_1a import question_1_a
from src.questions.question_1b import question_1_b
from src.questions.question_2a import question_2_a
from src.questions.question_2b import question_2_b
from src.questions.question_2c import question_2_c
from IPython.display import display
import pandas as pd
import json

def answer_question(question, pokemon_factory, config):
    match question:
        case "1-a":
            question_1_a(pokemon_factory, config)
        case "1-b":
            question_1_b(pokemon_factory, config)
        case "2-a":
            question_2_a(pokemon_factory, config)
        case "2-b":
            question_2_b(pokemon_factory, config)
        case "2-c":
            question_2_c(pokemon_factory, config)

if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    config = json.load(open("config.json"))
    snorlax = factory.create("snorlax", 100, StatusEffect.NONE, 1)
    while True:
        question = input("What question should be answered? [1-a|1-b|2-a|2-b|2-c]: ")
        answer_question(question, factory, config)
