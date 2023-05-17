import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from settings import settings

def get_dataset_principal_components(variables_data, n_components):
    """
    Get the principal components of a dataset
    :param n_components: The number of components
    :return: The principal components
	"""

    # Standardize the data
    # View documentation for more understanding
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(variables_data)

    # PCA
    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)

    return data_scaled, pca