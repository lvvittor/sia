import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def boxplot(variables_data, standardized=None):
	plt.figure(figsize=(10, 6))
	sns.boxplot(data=variables_data, palette='pastel')
	standardized = "Non " if not standardized else ""
	plt.title(f"{standardized}Standardized features")
	plt.show()


# FIXME: This function is not working properly
def biplot(countries, variables_data, pca):
	df = pd.DataFrame(data=variables_data, columns=[pca.components_[0], pca.components_[1]])
	df['Etiqueta'] = countries

	X_pca = pca.fit_transform(variables_data.T)

	# Crear el biplot utilizando seaborn
	plt.figure(figsize=(8, 6))
	plt.scatter(X_pca[:, 0], X_pca[:, 1], c='b', alpha=0.5)

	for i, label in enumerate(countries):
		plt.annotate(label, (X_pca[i, 0], X_pca[i, 1]))

	# Mostrar ejes de referencia
	for length, vector in zip(pca.explained_variance_, pca.components_):
		v = vector * 3 * np.sqrt(length)
		plt.arrow(0, 0, v[0], v[1], head_width=0.2, head_length=0.2, fc='r', ec='r')

	# Mostrar el biplot
	plt.grid()
	plt.show()
	