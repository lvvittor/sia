import numpy as np

from settings import settings

class MLP():
    """Multilayer perceptron class (1 hidden layer and 1 output layer)"""

    def __init__(
        self, layers: list[int], output_activation_func: str = "sigmoid"
    ):        
        self.learning_rate = settings.learning_rate

        # +1 in the column dim to account for bias term on each layer
        self.weights = [np.random.randn(layers[i + 1], layers[i] + 1) for i in range(len(layers) - 1)]

        self.output_activation_func = output_activation_func

        # Momentum optimization
        self.previous_deltas = [np.zeros(weight.shape) for weight in self.weights]

        # ADAM optimization
        self.m = [np.zeros((layers[i + 1], layers[i] + 1)) for i in range(len(layers) - 1)]
        self.v = [np.zeros((layers[i + 1], layers[i] + 1)) for i in range(len(layers) - 1)]


    def feed_forward(self, inputs):
        # Add neuron with constant output 1 to inputs, to account for bias
        X = np.insert(inputs, 0, 1, axis=1)

        V = [X.T]
        H = []

        # Iterate over hidden layers
        for i, w in enumerate(self.weights[:-1]):
            h = w.dot(V[i])
            v = self._relu(h)
            v = np.insert(v, 0, 1, axis=0) # add bias to hidden layer output
            H.append(h)
            V.append(v)

        hO = self.weights[-1].dot(V[-1]) # (35, 17) x (17, 32) = (35, 32)

        # Output layer output
        O = self._activation_func(hO, self.output_activation_func)

        return H, V, hO, O.T # omit inputs in V, transpose O to (N, output_nodes)


    def backward_propagation(self, epoch, H, V, hO, prev_delta_sum):
        # Update output layer weights
        dO = prev_delta_sum.T * self._activation_derivative(hO, self.output_activation_func)
        dW = self.learning_rate * dO.dot(V[-1].T)
        
        dV = dO
        dw = [dW]

        # Iterate backwards over hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            prev_layer_delta_sum = dV.T.dot(self.weights[i+1][:, 1:]) # remove bias from output layer weights
            dV = prev_layer_delta_sum.T * self._relu_derivative(H[i])
            # Note that V[i] is the output of the (i-1)-th hidden layer since V[0] is the input (and len(V) = len(H) + 1)
            dw.insert(0, self.learning_rate * dV.dot(V[i].T)) # insert at the beginning to keep the order

        prev_layer_delta_sum = dV.T.dot(self.weights[0][:, 1:]) # remove bias from hidden layer weights

        # Momentum optimization
        if settings.optimization == "momentum":
            for i in range(len(dw)):
                _dw = dw[i].copy()
                dw[i] -= 0.9 * self.previous_deltas[i]
                self.previous_deltas[i] = _dw
        # ADAM optimization
        elif settings.optimization == "adam":
            for i in range(len(dw)): # iterate over layers
                self.m[i] = settings.adam_optimization.beta1 * self.m[i] + (1 - settings.adam_optimization.beta1) * dw[i]
                self.v[i] = settings.adam_optimization.beta2 * self.v[i] + (1 - settings.adam_optimization.beta2) * (dw[i] ** 2)
                m_hat = self.m[i] / (1 - settings.adam_optimization.beta1**epoch)
                v_hat = self.v[i] / (1 - settings.adam_optimization.beta1**epoch)
                dw[i] = self.learning_rate * m_hat / (np.sqrt(v_hat) + settings.adam_optimization.epsilon)

        # Update weights
        for i in range(len(dw)):
            self.weights[i] -= dw[i]

        return prev_layer_delta_sum


    def _relu(self, V):
        return np.maximum(0, V)


    def _relu_derivative(self, V):
        return np.where(V > 0, 1, 0)


    def _sigmoid(self, V):
        restricted_V = np.clip(V, -500, 500) # avoid overflows
        return 1 / (1 + np.exp(-restricted_V))


    def _sigmoid_derivative(self, V):
        sigmoid_output = self._sigmoid(V)
        return sigmoid_output * (1 - sigmoid_output)

    
    def _linear(self, V):
        return V
    

    def _linear_derivative(self, V):
        return np.ones(V.shape)
    

    def _activation_func(self, V, function):
        match function:
            case "relu":
                return self._relu(V)
            case "sigmoid":
                return self._sigmoid(V)
            case "linear":
                return self._linear(V)
            case _:
                raise ValueError(f"Invalid activation function: {function}")


    def _activation_derivative(self, V, function):
        match function:
            case "relu":
                return self._relu_derivative(V)
            case "sigmoid":
                return self._sigmoid_derivative(V)
            case "linear":
                return self._linear_derivative(V)
            case _:
                raise ValueError(f"Invalid activation function: {function}")
