import numpy as np

from settings import settings

class MLP():
    """Multilayer perceptron class (1 hidden layer and 1 output layer)"""

    def __init__(
        self, input_dim: int, hidden_nodes: int, output_nodes: int, output_activation_func: str = "sigmoid"
    ):        
        self.learning_rate = settings.learning_rate

        # +1 for bias term on each layer
        self.weights = [
            np.random.randn(hidden_nodes, input_dim + 1),
            np.random.randn(output_nodes, hidden_nodes + 1)
        ]

        # Momentum optimization
        self.previous_deltas = [np.zeros(self.weights[0].shape), np.zeros(self.weights[1].shape)]

        self.output_activation_func = output_activation_func

        # Adam variables
        self.m_dw = np.zeros((hidden_nodes, input_dim + 1)) # Same size as dw in backward_propagation
        self.v_dw = np.zeros((hidden_nodes, input_dim + 1))
        self.m_dW = np.zeros((output_nodes, hidden_nodes + 1)) # Same size as dW in backward_propagation
        self.v_dW = np.zeros((output_nodes, hidden_nodes + 1))


    def feed_forward(self, inputs):
        # Add neuron with constant output 1 to inputs, to account for bias
        X = np.insert(inputs, 0, 1, axis=1)

        h1 = self.weights[0].dot(X.T)

        # Hidden layer output
        V1 = self._relu(h1)
        V1 = np.insert(V1, 0, 1, axis=0) # add bias to hidden layer output

        h2 = self.weights[1].dot(V1)

        # Output layer output
        O = self._activation_func(h2, self.output_activation_func)

        return h1, V1, h2, O.T # transpose to (N, output_nodes)


    def backward_propagation(self, inputs, h1, V1, h2, prev_delta_sum):
        output_errors = prev_delta_sum.T # (output_nodes, N)
        X = np.insert(inputs, 0, 1, axis=1) # add bias to inputs
        
        # Update output layer weights
        dO = output_errors * self._activation_derivative(h2, self.output_activation_func)
        dW = self.learning_rate * dO.dot(V1.T)
        
        # Update hidden layer weights
        output_layer_delta_sum = dO.T.dot(self.weights[1][:, 1:]) # remove bias from output layer weights
        dV1 = output_layer_delta_sum.T * self._relu_derivative(h1)
        dw = self.learning_rate * dV1.dot(X)

        hidden_layer_delta_sum = dV1.T.dot(self.weights[0][:, 1:]) # remove bias from hidden layer weights
        # Momentum optimization
        if settings.optimization == "momentum":
            self.previous_deltas = [dw.copy(), dW.copy()]
            dW -= 0.9 * self.previous_deltas[1]
            dw -= 0.9 * self.previous_deltas[0]
        # Adam optimization
        elif settings.optimization == "adam":
            self.m_dw = settings.adam_optimization.beta1 * self.m_dw + (1 - settings.adam_optimization.beta1) * dw
            self.m_dW = settings.adam_optimization.beta1 * self.m_dW + (1 - settings.adam_optimization.beta1) * dW
            self.v_dw = settings.adam_optimization.beta2 * self.v_dw + (1 - settings.adam_optimization.beta2) * (dw ** 2)
            self.v_dW = settings.adam_optimization.beta2 * self.v_dW + (1 - settings.adam_optimization.beta2) * (dW ** 2)

            m_dw_corrected = self.m_dw / (1 - settings.adam_optimization.beta1)
            m_dW_corrected = self.m_dW / (1 - settings.adam_optimization.beta1)
            v_dw_corrected = self.v_dw / (1 - settings.adam_optimization.beta2)
            v_dW_corrected = self.v_dW / (1 - settings.adam_optimization.beta2)

            dw = self.learning_rate * m_dw_corrected / (np.sqrt(v_dw_corrected) + settings.adam_optimization.epsilon)
            dW = self.learning_rate * m_dW_corrected / (np.sqrt(v_dW_corrected) + settings.adam_optimization.epsilon)

        self.weights[1] -= dW
        self.weights[0] -= dw

        return hidden_layer_delta_sum


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
