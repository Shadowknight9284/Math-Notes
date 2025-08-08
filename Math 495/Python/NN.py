import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activations):
        """
        Initialize network with:
        - layer_sizes: list of neurons per layer (e.g., [4, 16, 3])
        - activations: list of activation functions for hidden/output layers
        """
        self.layer_sizes = layer_sizes
        self.activations = [self._get_activation(fn) for fn in activations]
        self.params = self._initialize_parameters()

    def _initialize_parameters(self):
        params = {}
        for i in range(1, len(self.layer_sizes)):
            params[f'W{i}'] = np.random.randn(
                self.layer_sizes[i], self.layer_sizes[i-1]
            ) * np.sqrt(2./self.layer_sizes[i-1])  # He initialization
            params[f'b{i}'] = np.zeros((self.layer_sizes[i], 1))
        return params

    def _get_activation(self, name):
        activations = {
            'relu': (lambda x: np.maximum(0, x),
                     lambda x: (x > 0).astype(float)),
            'sigmoid': (lambda x: 1/(1 + np.exp(-x)),
                        lambda x: x*(1 - x)),
            'tanh': (lambda x: np.tanh(x),
                     lambda x: 1 - x**2),
            'linear': (lambda x: x,
                       lambda x: 1)
        }
        return activations[name]

    def forward(self, X):
        cache = {'A0': X.T}
        for i in range(1, len(self.layer_sizes)):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            Z = W @ cache[f'A{i-1}'] + b
            act_fn, _ = self.activations[i-1]
            cache[f'A{i}'] = act_fn(Z)
        return cache

    def backward(self, X, y, cache, learning_rate=0.01):
        m = X.shape[0]
        grads = {}
        
        # Output layer gradient
        AL = cache[f'A{len(self.layer_sizes)-1}']
        dZ = 2*(AL - y.T)  # MSE derivative
        for i in reversed(range(1, len(self.layer_sizes))):
            A_prev = cache[f'A{i-1}']
            W = self.params[f'W{i}']
            
            grads[f'dW{i}'] = (dZ @ A_prev.T) / m
            grads[f'db{i}'] = np.sum(dZ, axis=1, keepdims=True) / m
            
            if i > 1: # Propagate error backward
                _, act_deriv = self.activations[i-2]
                dZ = (W.T @ dZ) * act_deriv(A_prev)
        
        # Update parameters
        for i in range(1, len(self.layer_sizes)):
            self.params[f'W{i}'] -= learning_rate * grads[f'dW{i}']
            self.params[f'b{i}'] -= learning_rate * grads[f'db{i}']

    def train(self, X, y, epochs=1000, lr=0.01):
        for _ in range(epochs):
            cache = self.forward(X)
            self.backward(X, y, cache, learning_rate=lr)

    def predict(self, X):
        cache = self.forward(X)
        return cache[f'A{len(self.layer_sizes)-1}'].T
