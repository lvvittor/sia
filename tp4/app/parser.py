from settings import settings
import random
import numpy as np

class Parser:
    def __init__(self, base_path: str):
        self.base_path = base_path

    @property
    def letter_matrix(self):
        """Returns a matrix of letters, where each row is a letter."""
        matrix = []

        for letter in settings.hopfield.selected_letters:
            with open(f"{self.base_path}/{letter.lower()}.txt", "r") as f:
                parsed_letter = [[1 if char == '*' else -1 for char in line.strip("\n")] for line in f]
        
            matrix.append(parsed_letter)

        return matrix

    def apply_noise(self, parsed_letter: list):
        """Adds noise to a given letter."""
        return [[-value if random.random() < settings.hopfield.noise_level else value for value in row] for row in parsed_letter]

    def calculate_similarity(self, parsed_letter: list, parsed_letter_with_noise: list):
        """Calculates the similarity between two letters. 
            Returns the jaccard coefficient and a matrix with 1s where the letters are equal and 0s where they are different.
        """
        matrix1 = np.array(parsed_letter)
        matrix2 = np.array(parsed_letter_with_noise)

        intersection = np.sum(matrix1 == matrix2)
        union = matrix1.size + matrix2.size - intersection

        jaccard_coefficient = intersection / union

        element_wise_comparison = np.where(matrix1 == matrix2, 1, 0)

        return jaccard_coefficient, element_wise_comparison.tolist()

    

