import pandas as pd
import numpy as np

from settings import settings
from PCA import get_dataset_principal_components
from visualization import boxplot, biplot, component_barplot, country_heatmap, u_matrix, variable_value_scatter
from sklearn.preprocessing import StandardScaler

from kohonen import Kohonen
from oja import Oja

def main():
	# Parse dataset
	countries, variables_data = parse_dataset(f"{settings.Config.data_path}/europe.csv")
	variables = variables_data.to_numpy()
	standardized_vars = (variables - np.mean(variables, axis=0)) / np.std(variables, axis=0)

	# PCA with sklearn
	# pca_with_sklearn(countries, variables_data, 2)

	# Kohonen
	run_kohonen(countries, standardized_vars)

	# Oja
	# oja(countries, variables_data)


def oja(countries, variables_data):
	oja = Oja(variables_data.to_numpy())
	oja.train(500)

	# Visualize oja
	scaler = StandardScaler()
	standardized_data = scaler.fit_transform(variables_data)
	standardized_data = pd.DataFrame(data=standardized_data, columns=variables_data.columns.values)
	variables = ["Area", "GDP", "Inflation", "Life. expect", "Military", "Pop.growth", "Unemployment"]
	variable_value_scatter(variables, oja.weights)
	component_barplot(countries, standardized_data, oja.weights, "oja_barplot")
	print(oja.weights)

	_, pca = get_dataset_principal_components(variables_data, 2)
	print(pca.components_[0])

	print(np.dot(np.dot(oja.weights, np.cov(variables_data.to_numpy(), rowvar=False)),oja.weights.T))
	print(np.dot(np.dot(pca.components_[0], np.cov(variables_data.to_numpy(), rowvar=False)),pca.components_[0].T))


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


def run_kohonen(countries, dataset):
	""" Run kohonen algorithm with the given dataset, which MUST be standardized (mean=0, std=1) """
	k = 4
	epochs = 10_000

	kohonen = Kohonen(k, dataset)
	kohonen.train(epochs)

	winner_neurons = kohonen.map_inputs(dataset) # get the winner neuron for each input
	umatrix = kohonen.get_umatrix()			  	 # get u matrix

	country_heatmap(countries, winner_neurons, k)
	u_matrix(umatrix)


def parse_dataset(dataset: str):
	data = pd.read_csv(dataset)
	countries = data["Country"]
	variables_data = data.drop(columns=["Country"])
	return countries, variables_data


if __name__ == "__main__":
    main()
