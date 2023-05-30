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

    def rotate(self, parsed_letter: list):
        """Adds noise to a given letter."""
        return np.array([[value for value in row] for row in parsed_letter]).T

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

    def combinations(self, lst, n):
        if n == 0:
            yield []
        else:
            for i in range(len(lst)):
                rest = lst[i+1:]
                for c in self.combinations(rest, n-1):
                    yield [lst[i]] + c

    def find_orthogonal_columns(self, path):
        """ Calculates all posible combinations of orthogonal letters in the alphabet"""
        with open(path, 'r') as file:
            content = file.read()
        letters = [letter.split('\n') for letter in content.split('\n\n')]
        matrixes = [[[1 if char == '*' else -1 for char in row] for row in letter] for letter in letters]
        matrix = np.column_stack([pattern.flatten() for pattern in np.array(matrixes)])

        num_columns = matrix.shape[1]
        column_indices = np.arange(num_columns)
        orthogonal_sets = []

        for subset_indices in self.combinations(column_indices, 4):
            subset_matrix = matrix[:, subset_indices]
            dot_product_matrix = np.dot(subset_matrix.T, subset_matrix)
            if np.allclose(dot_product_matrix, np.eye(4), settings.hopfield.orthogonal_tolerance):
                orthogonal_sets.append(subset_indices)

        return orthogonal_sets

