import seaborn as sns
import matplotlib.pyplot as plt

from settings import settings

def boxplot(variables_data, standardized=None):
	plt.figure(figsize=(10, 6))
	sns.boxplot(data=variables_data, palette='pastel')
	standardized = "Non" if not standardized else ""
	plt.title(f"{standardized} Standardized features")
	plt.savefig(f"{settings.Config.output_path}/{standardized}StandardizedBoxplot.png")
	plt.show()


def biplot(countries, variables_data, pca, scaled = False):
	# PC1 and PC2 components of each country in the dataset
	pc_coords = pca.fit_transform(variables_data) # shape (28, 2)

	if scaled:
		scale_PC1 = 1.0 / (pc_coords[:, 0].max() - pc_coords[:, 0].min())
		scale_PC2 = 1.0 / (pc_coords[:, 1].max() - pc_coords[:, 1].min())
	else:
		scale_PC1 = 1.0
		scale_PC2 = 1.0

	plt.figure(figsize=(12, 6))

	# Plot dataset items with their PC1 and PC2 coordinates
	plt.scatter(pc_coords[:, 0] * scale_PC1, pc_coords[:, 1] * scale_PC2, c='b', alpha=0.5)

	# Add country name labels to each point in the plot
	for i, label in enumerate(countries):
		plt.annotate(label, (pc_coords[i, 0] * scale_PC1, pc_coords[i, 1] * scale_PC2), fontsize=7, alpha=0.75)

	# Show features as vectors, with their PC1 and PC2 coordinates
	for i, feature in enumerate(variables_data.columns.values):
		plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], head_width=0.03, head_length=0.03, color="red", alpha=0.6)
		# * 1.3 to make the text appear a bit further from the vector
		plt.text(pca.components_[0, i] * 1.15, pca.components_[1, i] * 1.15, feature, color="red", fontsize=9)

	plt.xlabel("PC1")
	plt.ylabel("PC2")

	plt.grid()
	plt.savefig(settings.Config.output_path+"/biplot.png")
	plt.show()
	

def component_barplot(countries, variables_data, component):
	values = variables_data.apply(lambda row: sum(row * component), axis=1)
	plt.figure(figsize=(12, 6))

	sns.set(style="whitegrid")
	ax = sns.barplot(x=countries, y=values)
	ax.set(xlabel='', ylabel='PC1', title='')

	plt.xticks(rotation=90)
	plt.tight_layout()

	plt.savefig(settings.Config.output_path+"/pc1_barplot.png")
	plt.show()