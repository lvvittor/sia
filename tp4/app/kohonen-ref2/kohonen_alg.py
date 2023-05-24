import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

class Kohonen:

    def __init__(self,p, n, k,radio, learning_rate, similitud, epochs, X,country_name_train, categories):
        self.p = p
        self.n = n
        self.k = k
        self.neurons = np.zeros((k,k))
        self.neurons_reshape = self.neurons.reshape(k**2)

        self.weights = []

        # Initialize weights with samples from X (inputs)
        for _ in range(k**2):
            index = np.random.randint(0, p-1)
            if(n==1):
                x = self.standard_i(X)
            else:
                x = self.standard_i(X[index])
            self.weights.append(x) 

        self.radio = [radio,radio]
        self.learning_rate = [learning_rate,learning_rate]
        self.similitud = similitud
        self.epochs = epochs
        self.X = X
        self.country_name_train = country_name_train
        self.categories = categories


    def train_kohonen(self):
        print("NON STANDARD DATA shape:", np.array(self.X).shape)
        print(self.X)

        X_standard = self.standard(self.X)

        print("STANDARD DATA shape:", np.array(X_standard).shape)
        print(X_standard)

        # Winner index de cada registro de entrada
        neuron_country = np.zeros(len(X_standard))

        for i in range(self.epochs):
            for j in range(len(X_standard)): # TODO: try
                # Seleccionar un registro de entrada X^p
                x = X_standard[j]
                # Encontrar la neurona ganadora
                winner_index = self.winner(x)

                # Get neighborhood of winner neuron
                distances = self.activation(winner_index, i)
                # Actualizar los pesos segun kohonen
                self.regla_de_kohonen(distances, x)

                neuron_country[j] = winner_index

            # Ajuste de radio:
            ajuste = self.radio[0] * (1 - i/self.epochs)
            self.radio[1] = 1 if ajuste < 1 else ajuste # TODO: try

            # Ajuste de ETA:
            self.learning_rate[1] = self.learning_rate[0] * (1 - i/self.epochs) # TODO: try

        self.neurons = self.neurons_reshape.reshape(self.k,self.k)

        return neuron_country
    

    def standard(self, X):
        X_standard = []
        for i in range(len(X)):
            X_standard.append(self.standard_i(X[i]))    
        return X_standard
    
    
    def standard_i(self,x):
        mean =[np.mean(x) for _ in range(len(x))]
        std = [np.std(x) for _ in range(len(x))]
        return (x - np.array(mean))/np.array(std) 

    
    def regla_de_kohonen(self, distances, x):
        for j in range(self.k**2):
            # Si soy vecino actulizo mis pesos
            if(j in distances):
                for p in range(self.n):
                    self.weights[j][p] += self.learning_rate[1] * (x[p]-self.weights[j][p])


    def winner(self, x):
        if(self.similitud == "euclidea"):
            return self.euclidea(x)
        else:
            return self.exponencial(x)


    def euclidea(self, x):
        w=[]
        for j in range(self.k**2): #recorriendo filas
            w.append(np.linalg.norm(x - self.weights[j]))
        wk = min(w)
        return w.index(wk)
    

    def exponencial(self, x):
        w=[]
        for j in range(self.k**2): #recorriendo filas
            w.append(np.exp(-((np.linalg.norm(x - self.weights[j]))**2)))
        wk = min(w)
        return w.index(wk)
    

    def activation(self, winner_index, epoch):
        if(epoch == self.epochs - 1):
            self.neurons_reshape[winner_index] += 1
            self.neurons = self.neurons_reshape.reshape(self.k,self.k)

        winner_pos = np.unravel_index(winner_index, self.neurons.shape)
        distances = []
        # Obtengo la distancia entre neuronas
        for i in range(self.k):
            for j in range(self.k):
                distance = self.get_neighbours_distance(np.array(winner_pos), [i,j])
                # Veo si son vecinos
                if(distance <= self.radio[1]):
                    distances.append(np.ravel_multi_index((i, j), self.neurons.shape))

        return distances
    

    def get_neighbours_distance(self, winner_pos, neurons):
        #winner_pos = [x,y]
        #neurons = [a,b]
        return np.linalg.norm(winner_pos - neurons)
    

    def plot_heatmap(self, similitud, neurons_countries):
        fig, ax = plt.subplots(1, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "orange", "red"])
        im = ax.imshow(self.neurons, cmap=cmap)

        for j in range(self.k**2):
            winner_pos = np.array(np.unravel_index(j, self.neurons.shape))
            country_label = ""
            for idx in range(self.p):
                if(neurons_countries[idx] == j):
                    country_label = country_label + self.country_name_train[idx] + '\n'
            ax.text(winner_pos[1], winner_pos[0], country_label, ha="center", va="center", color="black", fontsize=5)

        fig.colorbar(im)
        plt.title(f'Grilla de neuronas de {self.k}x{self.k} con similitud {similitud}')
        ax.yaxis.set_major_locator(plt.NullLocator())  # remove y axis ticks
        ax.xaxis.set_major_locator(plt.NullLocator())  # remove x axis ticks
        plt.show()


    def plot_category(self, categoryIdx, neurons_countries):
        train_category = [fila[categoryIdx] for fila in self.X]
        fig, ax = plt.subplots(1, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "yellow", "green", "blue"])
        avg_matriz = np.zeros((self.k,self.k))
        for j in range(self.k**2):
            winner_pos = np.array(np.unravel_index(j, self.neurons.shape))
            country_label = ""
            avg_j = []
            for idx in range(self.p):
                if(neurons_countries[idx] == j):
                    avg_j.append(train_category[idx])
                    country_label = country_label + self.country_name_train[idx]
            avg_matriz[winner_pos[1], winner_pos[0]] = np.mean(avg_j)
        im = ax.imshow(avg_matriz, cmap=cmap)
        fig.colorbar(im)
        plt.title(f'Grilla de neuronas de {self.k}x{self.k} para categoria: {self.categories[categoryIdx]}')
        ax.yaxis.set_major_locator(plt.NullLocator())  # remove y axis ticks
        ax.xaxis.set_major_locator(plt.NullLocator())  # remove x axis ticks
        plt.show()


    def plot_u_matrix(self, similitud):
        fig, ax = plt.subplots(1, 1)
        distances = np.zeros(shape=(self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                distances[i][j] = self.get_neighbours_weight_distance([i, j], similitud)
                # ax.text(j, i, distances[i][j] , ha="center", va="center", color="red", fontsize=5)

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "grey", "black"])
        im = ax.imshow(distances, cmap=cmap)
        fig.colorbar(im)
        plt.title(f'Media de distancia {similitud} entre pesos de neuronas vecinas')
        ax.yaxis.set_major_locator(plt.NullLocator())  # remove y axis ticks
        ax.xaxis.set_major_locator(plt.NullLocator())  # remove x axis ticks
        plt.show()


    def get_neighbours_weight_distance(self, winner_pos, similitud):
        distances=[]
        for i in range(self.k):
            for j in range(self.k):
                distance = self.get_neighbours_distance(np.array(winner_pos), [i,j])
                # Veo si son vecinos
                if(distance <= self.radio[1]):
                    n_idx = np.ravel_multi_index((i, j), self.neurons.shape)
                    w_idx = np.ravel_multi_index((winner_pos[0], winner_pos[1]), self.neurons.shape)
                    # distances.append(distance)
                    if(similitud == 'euclidea'):
                        distances.append(np.linalg.norm(self.weights[w_idx] - self.weights[n_idx]))
                    else:
                        distances.append(np.exp(-((np.linalg.norm(self.weights[w_idx] - self.weights[n_idx]))**2)))
        return np.mean(distances)
