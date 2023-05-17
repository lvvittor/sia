import pandas as pd

from settings import settings
from PCA import get_dataset_principal_components
from visualization import boxplot, biplot

def main():
	countries, variables_data = parse_dataset(f"{settings.Config.data_path}/europe.csv")
	
	pca_with_sklearn(countries, variables_data, 2)


def pca_with_sklearn(countries, variables_data, n_components):
	boxplot(variables_data)

	standardized_data, pca = get_dataset_principal_components(variables_data, n_components)
	# Add column names and index
	standardized_data = pd.DataFrame(data=standardized_data, columns=variables_data.columns.values)
    
	print("PC1: ", pca.components_[0])
	print("PC2: ", pca.components_[1])

	boxplot(standardized_data, standardized=True)

	# FIXME: This function is not working properly
	biplot(countries, standardized_data, pca)
        

def parse_dataset(dataset: str):
	data = pd.read_csv(dataset)
	countries = data["Country"]
	variables_data = data.drop(columns=["Country"])
	return countries, variables_data


if __name__ == "__main__":
    main()
