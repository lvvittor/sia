import pandas as pd

from settings import settings
from PCA import get_dataset_principal_components
from visualization import boxplot, biplot, component_barplot

def main():
	countries, variables_data = parse_dataset(f"{settings.Config.data_path}/europe.csv")
	
	pca_with_sklearn(countries, variables_data, 2)


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
