from __future__ import annotations 

import pandas as pd
import random as r

from settings import settings
from PCA import get_dataset_principal_components
from sklearn.preprocessing import StandardScaler
from visualization import boxplot, biplot, component_barplot
from kohonen import Kohonen
from discrete_hopfield import DiscreteHopfield
from parser import Parser
from oja import Oja
from pprint import pprint

def main():
	match settings.exercise:
		case 1:
			#countries, variables_data = parse_dataset(f"{settings.Config.data_path}/europe.csv")

			# pca_with_sklearn(countries, variables_data, 2)
			# kohonen = Kohonen(4, variables_data.to_numpy())
			# kohonen.train()
			pass
		case 2:
			hopfield()


def oja(countries, variables_data):
	oja = Oja(variables_data.to_numpy())
	oja.train(500)

	# Visualize oja
	scaler = StandardScaler()
	standardized_data = scaler.fit_transform(variables_data)
	standardized_data = pd.DataFrame(data=standardized_data, columns=variables_data.columns.values)
	component_barplot(countries, standardized_data, oja.weights, "oja_barplot")
	print(oja.weights)


def hopfield():
	# Parse letters into a 5x5 matrix of 1s and -1s 
	parser = Parser(f"{settings.Config.data_path}/letters")
	
	print("Parsed letters:")
	pprint(parser.letter_matrix)

	print(f"Similarity comparison with XI and ZETA: (jaccard_coefficient, element_wise_comparison)")

	for parsed_letter in parser.letter_matrix:
		parsed_letter_with_noise = parser.apply_noise(parsed_letter)
		pprint(parser.calculate_similarity(parsed_letter, parsed_letter_with_noise))


	hopfield = DiscreteHopfield()


def pca_with_sklearn(countries, variables_data, n_components):
	boxplot(variables_data)

	standardized_data, pca = get_dataset_principal_components(variables_data, n_components)
	# Add column names and index
	standardized_data = pd.DataFrame(data=standardized_data, columns=variables_data.columns.values)
    
	feature_charges = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=variables_data.columns.values)
	print(feature_charges)

	boxplot(standardized_data, standardized=True)

	# biplot(countries, standardized_data, pca, scaled=True)
	biplot(countries, standardized_data, pca)

	component_barplot(countries, standardized_data, pca.components_[0], "pc1_barplot")
        

def parse_dataset(dataset: str):
	data = pd.read_csv(dataset)
	countries = data["Country"]
	variables_data = data.drop(columns=["Country"])
	return countries, variables_data


if __name__ == "__main__":
    main()
