import seaborn as sns
import matplotlib.pyplot as plt

def boxplot(variables_data, standardized=None):
	plt.figure(figsize=(10, 6))
	sns.boxplot(data=variables_data, palette='pastel')
	standardized = "Non " if not standardized else ""
	plt.title(f"{standardized}Standardized features")
	plt.show()


def biplot(countries, variables_data, pca):
	# PC1 and PC2 components of each country in the dataset
	pc_coords = pca.fit_transform(variables_data) # shape (28, 2)

	plt.figure(figsize=(12, 6))

	# Plot dataset items with their PC1 and PC2 coordinates
	plt.scatter(pc_coords[:, 0], pc_coords[:, 1], c='b', alpha=0.5)

	# Add country name labels to each point in the plot
	for i, label in enumerate(countries):
		plt.annotate(label, (pc_coords[i, 0], pc_coords[i, 1]), fontsize=7, alpha=0.75)

	# Show features as vectors, with their PC1 and PC2 coordinates
	for i, feature in enumerate(variables_data.columns.values):
		plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], head_width=0.03, head_length=0.03, color="red", alpha=0.6)
		# * 1.3 to make the text appear a bit further from the vector
		plt.text(pca.components_[0, i] * 1.3, pca.components_[1, i] * 1.3, feature, color="red", fontsize=9)

	plt.xlabel("PC1")
	plt.ylabel("PC2")

	plt.grid()
	plt.show()
	