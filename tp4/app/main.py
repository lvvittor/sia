import pandas as pd

from settings import settings
from PCA import get_dataset_principal_components
from visualization import boxplot, biplot, component_barplot, country_heatmap
from kohonen import Kohonen

def main():
	# Parse dataset
	countries, variables_data = parse_dataset(f"{settings.Config.data_path}/europe.csv")

	# PCA with sklearn
	# pca_with_sklearn(countries, variables_data, 2)


	# Kohonen
	variables = variables_data.to_numpy()
	k = 4

	kohonen = Kohonen(k, variables)
	kohonen.train(1_000)

	winner_neurons = kohonen.map_inputs(variables) # get the winner neuron for each input
	heatmap = kohonen.get_heatmap(variables)	   # get amount of inputs mapped to each neuron

	# print("\n\n---WINNER NEURONS---")
	# print(winner_neurons)
	print("---HEATMAP---")
	print(heatmap)

	country_heatmap(countries, winner_neurons, k)


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

	component_barplot(countries, standardized_data, pca.components_[0])
        

def parse_dataset(dataset: str):
	data = pd.read_csv(dataset)
	countries = data["Country"]
	variables_data = data.drop(columns=["Country"])
	return countries, variables_data


if __name__ == "__main__":
    main()
