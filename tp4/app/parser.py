from settings import settings
import string

class Parser:
    def __init__(path: str):
        self.path = path # data/alphabet.txt
        self.letters = Parser.read_letters_from_file(self.path)
        self.letter_matrix = Parser.create_letter_matrix(self.letters)

    @property
    def selected_letters_matrix(self):
        """Returns a list of matrices for the selected letters"""
        return [self.letter_matrix[string.lowercase.index(selected_letter)] for selected_letter in settings.selected_letters]

    @classmethod
    def convert_to_matrix(cls, letter):
        """Converts a letter to a matrix of 1s and -1s"""
        return [[1 if char == '*' else -1 for char in row] for row in letter]

    @classmethod
    def create_letter_matrix(cls, letters):
        """Creates a matrix of 1s and -1s for each letter"""
        return [Parser.convert_to_matrix(letter) for letter in letters]

    @classmethod
    def read_letters_from_file(cls, path):
        """Reads the letters from a file and returns a list of letters"""
        with open(path, 'r') as file:
            content = file.read()
        return [letter.split('\n') for letter in content.split('\n\n')]

