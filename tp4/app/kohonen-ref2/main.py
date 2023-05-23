import numpy as np

import utils
import kohonen_alg

if __name__ == '__main__':
    epochs = 1000
    learning_rate = 0.5

    input_names, inputs, categories = utils.import_data('tp4/data/europe.csv')
    country_name_train = np.array(input_names)
    training_set = np.array(inputs, dtype=float)

    p = len(training_set)
    n = len(training_set[0])

    radio = 8
    similitud = "euclidea"
    k = 4
    
    model = kohonen_alg.Kohonen(p, n, k, radio, learning_rate, similitud, epochs,training_set,country_name_train, categories)
    neurons_countries = model.train_kohonen()
    model.plot_heatmap(similitud, neurons_countries)

    # Categories Heatmap
    for categoryIdx in range(len(categories)):
        model.plot_category(categoryIdx, neurons_countries)

    # Matriz U
    model.plot_u_matrix(similitud)
