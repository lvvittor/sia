import pandas as pd

from settings import settings
from PCA import get_dataset_principal_components
from visualization import boxplot, biplot, component_barplot, country_heatmap, u_matrix
from kohonen import Kohonen

def main():
	# Parse dataset
	countries, variables_data = parse_dataset(f"{settings.Config.data_path}/europe.csv")

	# PCA with sklearn
	# pca_with_sklearn(countries, variables_data, 2)

	#  Convert dataset to numpy array
	variables = variables_data.to_numpy()

	# Kohonen
	run_kohonen(countries, variables)


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


def run_kohonen(countries, dataset):
	k = 4
	epochs = 1_000

	kohonen = Kohonen(k, dataset)
	kohonen.train(epochs)

	winner_neurons = kohonen.map_inputs(dataset) # get the winner neuron for each input
	umatrix = kohonen.get_umatrix()			  	   # get u matrix

	country_heatmap(countries, winner_neurons, k)
	u_matrix(umatrix)


def parse_dataset(dataset: str):
	data = pd.read_csv(dataset)
	countries = data["Country"]
	variables_data = data.drop(columns=["Country"])
	return countries, variables_data


if __name__ == "__main__":
    main()
