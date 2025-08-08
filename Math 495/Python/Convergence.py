import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns

class PCA:
    """
    Principal Component Analysis implementation with adjustable components
    that outputs explained variance ratios.
    """
    def __init__(self, n_components=None):
        """
        Initialize PCA.
        
        Parameters:
        -----------
        n_components : int or None
            Number of principal components to keep. If None, all components are kept.
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.n_samples_ = None
        self.n_features_ = None
        
    def fit(self, X):
        """
        Fit PCA to data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        self.n_samples_, self.n_features_ = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Get variance explained by singular values
        explained_variance = (S ** 2) / (self.n_samples_ - 1)
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var
        
        # Adjust number of components if needed
        if self.n_components is None:
            self.n_components = min(self.n_samples_, self.n_features_)
        
        # Store results
        self.singular_values_ = S[:self.n_components]
        self.components_ = Vt[:self.n_components]
        self.explained_variance_ = explained_variance[:self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components]
        
        return self
    
    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform.
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        X_centered = X - self.mean_
        X_transformed = np.dot(X_centered, self.components_.T)
        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit PCA to data and apply dimensionality reduction.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Transform data back to original space.
        
        Parameters:
        -----------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data.
            
        Returns:
        --------
        X_reconstructed : array-like, shape (n_samples, n_features)
            Reconstructed data.
        """
        X_reconstructed = np.dot(X_transformed, self.components_) + self.mean_
        return X_reconstructed

class Activation:
    """Activation functions for neural networks."""
    
    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x, derivative=False):
        s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
        if derivative:
            return s * (1 - s)
        return s
    
    @staticmethod
    def tanh(x, derivative=False):
        t = np.tanh(x)
        if derivative:
            return 1 - t**2
        return t
    
    @staticmethod
    def linear(x, derivative=False):
        if derivative:
            return np.ones_like(x)
        return x
    
    @staticmethod
    def softmax(x, derivative=False):
        # Subtract max for numerical stability
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        s = e_x / np.sum(e_x, axis=1, keepdims=True)
        if derivative:
            # This is a simplification that works when used with cross-entropy loss
            return np.ones_like(s)
        return s
    
    @classmethod
    def get_activation(cls, name):
        """Get activation function by name."""
        activations = {
            'relu': cls.relu,
            'sigmoid': cls.sigmoid,
            'tanh': cls.tanh,
            'linear': cls.linear,
            'softmax': cls.softmax
        }
        
        if name.lower() not in activations:
            raise ValueError(f"Activation function '{name}' not supported")
        
        return activations[name.lower()]

class Layer:
    """Layer in a neural network."""
    
    def __init__(self, input_size, output_size, activation='relu'):
        """
        Initialize a layer with random weights and zeros as biases.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        output_size : int
            Number of output features (neurons in this layer)
        activation : str
            Activation function name
        """
        # He initialization for ReLU, Xavier for others
        if activation.lower() == 'relu':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        else:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
            
        self.biases = np.zeros((1, output_size))
        self.activation_name = activation.lower()
        self.activation_func = Activation.get_activation(activation)
        
        # For backpropagation
        self.input = None
        self.output = None
        self.input_before_activation = None
        
    def forward(self, input_data):
        """
        Forward pass through layer.
        
        Parameters:
        -----------
        input_data : numpy.ndarray
            Input data to the layer
            
        Returns:
        --------
        numpy.ndarray
            Output after activation
        """
        self.input = input_data
        self.input_before_activation = np.dot(input_data, self.weights) + self.biases
        self.output = self.activation_func(self.input_before_activation)
        return self.output
    
    def backward(self, output_error, learning_rate, momentum=0.9):
        """
        Backward pass through layer.
        
        Parameters:
        -----------
        output_error : numpy.ndarray
            Error from the next layer
        learning_rate : float
            Learning rate for gradient descent
        momentum : float, optional
            Momentum factor for faster convergence
            
        Returns:
        --------
        numpy.ndarray
            Error to propagate to the previous layer
        """
        # Calculate error derivative with respect to output
        delta = output_error * self.activation_func(self.input_before_activation, derivative=True)
        
        # Calculate error for previous layer
        input_error = np.dot(delta, self.weights.T)
        
        # Calculate gradients for weights and biases
        weights_gradient = np.dot(self.input.T, delta)
        biases_gradient = np.sum(delta, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        
        return input_error

class NeuralNetwork:
    """
    Neural network with customizable architecture and activation functions.
    """
    
    def __init__(self, input_size, hidden_layers=None, output_size=None, activation='relu', output_activation='softmax'):
        """
        Initialize a neural network.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_layers : list of int, optional
            List with the number of neurons in each hidden layer
        output_size : int, optional
            Number of output neurons (classes for classification)
        activation : str, optional
            Activation function name for hidden layers
        output_activation : str, optional
            Activation function name for output layer
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers or []
        self.output_size = output_size or 1
        self.activation = activation
        self.output_activation = output_activation
        self.layers = []
        
        # Build the network
        self._build_network()
        
        # For tracking metrics
        self.loss_history = []
        self.accuracy_history = []
        
    def _build_network(self):
        """Build the neural network architecture."""
        # Input to first hidden layer (or directly to output if no hidden layers)
        if not self.hidden_layers:
            self.layers.append(Layer(self.input_size, self.output_size, self.output_activation))
            return
        
        # Input to first hidden layer
        self.layers.append(Layer(self.input_size, self.hidden_layers[0], self.activation))
        
        # Hidden layers
        for i in range(len(self.hidden_layers) - 1):
            self.layers.append(Layer(self.hidden_layers[i], self.hidden_layers[i+1], self.activation))
        
        # Last hidden layer to output
        self.layers.append(Layer(self.hidden_layers[-1], self.output_size, self.output_activation))
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Network output
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def _compute_loss(self, y_pred, y_true):
        """
        Compute loss for current prediction.
        
        Parameters:
        -----------
        y_pred : numpy.ndarray
            Predicted values
        y_true : numpy.ndarray
            True values
            
        Returns:
        --------
        float
            Mean square error
        """
        # For binary classification or regression
        if self.output_size == 1:
            return np.mean((y_pred - y_true) ** 2)
        
        # For multi-class classification with one-hot encoding
        # Cross-entropy loss
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def _compute_loss_gradient(self, y_pred, y_true):
        """
        Compute gradient of loss with respect to predictions.
        
        Parameters:
        -----------
        y_pred : numpy.ndarray
            Predicted values
        y_true : numpy.ndarray
            True values
            
        Returns:
        --------
        numpy.ndarray
            Loss gradient
        """
        # For binary classification or regression
        if self.output_size == 1:
            return 2 * (y_pred - y_true) / y_true.shape[0]
        
        # For multi-class classification with one-hot encoding
        # Derivative of cross-entropy with softmax
        return (y_pred - y_true) / y_true.shape[0]
    
    def backward(self, y_pred, y_true, learning_rate):
        """
        Backward pass to update weights.
        
        Parameters:
        -----------
        y_pred : numpy.ndarray
            Predicted values
        y_true : numpy.ndarray
            True values
        learning_rate : float
            Learning rate for gradient descent
        """
        error = self._compute_loss_gradient(y_pred, y_true)
        
        # Backpropagate through layers in reverse order
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)
    
    def fit(self, X, y, epochs=100, batch_size=32, learning_rate=0.01, validation_data=None, verbose=1):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training data
        y : numpy.ndarray
            Target values
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Batch size for mini-batch gradient descent
        learning_rate : float, optional
            Learning rate for gradient descent
        validation_data : tuple, optional
            (X_val, y_val) for tracking validation metrics
        verbose : int, optional
            0: silent, 1: progress bar, 2: one line per epoch
            
        Returns:
        --------
        self
        """
        n_samples = X.shape[0]
        
        # Convert y to one-hot encoding if needed (multiclass)
        if self.output_size > 1 and y.ndim == 1:
            y_onehot = np.zeros((n_samples, self.output_size))
            y_onehot[np.arange(n_samples), y.astype(int)] = 1
            y = y_onehot
        
        # Reshape y for binary classification or regression if needed
        if self.output_size == 1 and y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self.loss_history = []
        self.accuracy_history = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Backward pass
                self.backward(y_pred, y_batch, learning_rate)
            
            # Calculate loss and accuracy for entire dataset
            y_pred_all = self.forward(X)
            loss = self._compute_loss(y_pred_all, y)
            self.loss_history.append(loss)
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(y_pred_all, y)
            self.accuracy_history.append(accuracy)
            
            # Validation metrics
            val_loss, val_accuracy = None, None
            if validation_data is not None:
                X_val, y_val = validation_data
                
                # Convert validation y to one-hot encoding if needed
                if self.output_size > 1 and y_val.ndim == 1:
                    y_val_onehot = np.zeros((len(y_val), self.output_size))
                    y_val_onehot[np.arange(len(y_val)), y_val.astype(int)] = 1
                    y_val = y_val_onehot
                
                # Reshape y_val for binary classification or regression if needed
                if self.output_size == 1 and y_val.ndim == 1:
                    y_val = y_val.reshape(-1, 1)
                
                y_val_pred = self.forward(X_val)
                val_loss = self._compute_loss(y_val_pred, y_val)
                val_accuracy = self._calculate_accuracy(y_val_pred, y_val)
            
            # Print progress
            if verbose == 1 and epoch % (epochs // 10 or 1) == 0:
                if validation_data is not None:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {accuracy:.4f}")
            elif verbose == 2:
                if validation_data is not None:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {accuracy:.4f}")
        
        return self
    
    def _calculate_accuracy(self, y_pred, y_true):
        """Calculate accuracy for current predictions."""
        if self.output_size == 1:  # Binary classification
            y_pred_class = (y_pred > 0.5).astype(int)
            return np.mean(y_pred_class == y_true)
        else:  # Multi-class classification
            y_pred_class = np.argmax(y_pred, axis=1)
            if y_true.ndim > 1:  # If one-hot encoded
                y_true_class = np.argmax(y_true, axis=1)
            else:
                y_true_class = y_true
            return np.mean(y_pred_class == y_true_class)
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        y_pred = self.forward(X)
        
        # For binary classification or regression
        if self.output_size == 1:
            return y_pred
        
        # For multi-class classification
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Predicted probabilities
        """
        return self.forward(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Test data
        y : numpy.ndarray
            True values
            
        Returns:
        --------
        tuple
            (loss, accuracy)
        """
        # Convert y to one-hot encoding if needed (multiclass)
        if self.output_size > 1 and y.ndim == 1:
            y_onehot = np.zeros((len(y), self.output_size))
            y_onehot[np.arange(len(y)), y.astype(int)] = 1
            y = y_onehot
        
        # Reshape y for binary classification or regression if needed
        if self.output_size == 1 and y.ndim == 1:
            y = y.reshape(-1, 1)
        
        y_pred = self.forward(X)
        loss = self._compute_loss(y_pred, y)
        accuracy = self._calculate_accuracy(y_pred, y)
        
        return loss, accuracy

class PCAClassifier:
    """
    PCA-based classifier that reduces dimensionality and uses logistic regression.
    """
    
    def __init__(self, n_components=None):
        """
        Initialize PCA classifier.
        
        Parameters:
        -----------
        n_components : int or None
            Number of principal components to keep. If None, 95% variance is kept.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.classifier = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """
        Fit PCA and classifier.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training data
        y : numpy.ndarray
            Target values
            
        Returns:
        --------
        self
        """
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        X_reduced = self.pca.fit_transform(X_scaled)
        
        # Fit classifier
        self.classifier.fit(X_reduced, y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Predicted class labels
        """
        X_scaled = self.scaler.transform(X)
        X_reduced = self.pca.transform(X_scaled)
        return self.classifier.predict(X_reduced)
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Predicted probabilities
        """
        X_scaled = self.scaler.transform(X)
        X_reduced = self.pca.transform(X_scaled)
        return self.classifier.predict_proba(X_reduced)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Test data
        y : numpy.ndarray
            True values
            
        Returns:
        --------
        float
            Accuracy score
        """
        return accuracy_score(y, self.predict(X))

class ModelComparison:
    """
    Framework for comparing different models.
    """
    
    def __init__(self, models, model_names=None):
        """
        Initialize comparison framework.
        
        Parameters:
        -----------
        models : list
            List of models to compare
        model_names : list of str, optional
            Names of the models
        """
        self.models = models
        self.model_names = model_names or [f"Model {i+1}" for i in range(len(models))]
        self.results = {}
        
    def compare(self, X, y, test_size=0.2, data_fractions=None, n_repeats=1, random_state=None):
        """
        Compare models with increasing training data size.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features
        y : numpy.ndarray
            Target variable
        test_size : float, optional
            Fraction of data to use for testing
        data_fractions : list of float, optional
            Fractions of training data to use (0.1 to 1.0)
        n_repeats : int, optional
            Number of times to repeat the experiment for statistical significance
        random_state : int or None, optional
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Results of the comparison
        """
        if data_fractions is None:
            data_fractions = np.linspace(0.1, 1.0, 10)
        
        # Split data into training and testing sets
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Initialize results dictionary
        results = {
            'data_fractions': data_fractions,
            'train_sizes': [int(f * len(X_train_full)) for f in data_fractions],
            'accuracy': {name: np.zeros((n_repeats, len(data_fractions))) for name in self.model_names},
            'training_time': {name: np.zeros((n_repeats, len(data_fractions))) for name in self.model_names},
            'model_instances': {name: [] for name in self.model_names}
        }
        
        # For each repeat
        for repeat in range(n_repeats):
            print(f"Repeat {repeat+1}/{n_repeats}")
            
            # Shuffle training data
            indices = np.random.permutation(len(X_train_full))
            X_train_full_shuffled = X_train_full[indices]
            y_train_full_shuffled = y_train_full[indices]
            
            # For each data fraction
            for i, fraction in enumerate(data_fractions):
                train_size = int(fraction * len(X_train_full))
                X_train = X_train_full_shuffled[:train_size]
                y_train = y_train_full_shuffled[:train_size]
                
                print(f"  Data fraction: {fraction:.1f} (train size: {train_size})")
                
                # For each model
                for j, (name, model_class) in enumerate(zip(self.model_names, self.models)):
                    print(f"    Training {name}...")
                    
                    # Create and train model
                    if isinstance(model_class, type):
                        model = model_class()
                    else:
                        # Clone the model if it's already instantiated
                        model = model_class
                    
                    # Train model and measure time
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    train_time = time.time() - start_time
                    
                    # Evaluate model
                    accuracy = accuracy_score(y_test, model.predict(X_test))
                    
                    # Record results
                    results['accuracy'][name][repeat, i] = accuracy
                    results['training_time'][name][repeat, i] = train_time
                    
                    # Save model instance (for the last repeat only)
                    if repeat == n_repeats - 1:
                        results['model_instances'][name].append(model)
                    
                    print(f"      Accuracy: {accuracy:.4f}, Training time: {train_time:.4f}s")
        
        # Calculate mean and std of accuracy and training time across repeats
        for name in self.model_names:
            results[f'{name}_accuracy_mean'] = np.mean(results['accuracy'][name], axis=0)
            results[f'{name}_accuracy_std'] = np.std(results['accuracy'][name], axis=0)
            results[f'{name}_training_time_mean'] = np.mean(results['training_time'][name], axis=0)
            results[f'{name}_training_time_std'] = np.std(results['training_time'][name], axis=0)
        
        # Find crossover point
        crossover_points = self._find_crossover_points(results)
        results['crossover_points'] = crossover_points
        
        # Calculate efficiency metrics
        results['efficiency_metrics'] = self._calculate_efficiency_metrics(results)
        
        # Store results
        self.results = results
        
        return results
    
    def _find_crossover_points(self, results):
        """Find points where performance of models cross each other."""
        crossover_points = {}
        model_pairs = [(i, j) for i in range(len(self.model_names)) for j in range(i+1, len(self.model_names))]
        
        for i, j in model_pairs:
            name_i, name_j = self.model_names[i], self.model_names[j]
            accuracy_i = results[f'{name_i}_accuracy_mean']
            accuracy_j = results[f'{name_j}_accuracy_mean']
            
            # Check for crossover
            diff = accuracy_i - accuracy_j
            crossovers = []
            for k in range(len(diff) - 1):
                if diff[k] * diff[k+1] <= 0:  # Sign change or one value is zero
                    # Linear interpolation to find exact crossover point
                    if diff[k] == diff[k+1] == 0:
                        crossover_fraction = results['data_fractions'][k]
                    elif diff[k] == 0:
                        crossover_fraction = results['data_fractions'][k]
                    elif diff[k+1] == 0:
                        crossover_fraction = results['data_fractions'][k+1]
                    else:
                        w1 = abs(diff[k+1]) / (abs(diff[k]) + abs(diff[k+1]))
                        w2 = abs(diff[k]) / (abs(diff[k]) + abs(diff[k+1]))
                        crossover_fraction = w1 * results['data_fractions'][k] + w2 * results['data_fractions'][k+1]
                    
                    crossover_size = int(crossover_fraction * len(results['train_sizes']))
                    
                    # Which model is better after crossover
                    if k+2 < len(diff) and diff[k+2] > 0:
                        better_after = name_i
                    else:
                        better_after = name_j
                    
                    crossovers.append({
                        'data_fraction': crossover_fraction,
                        'train_size_approx': crossover_size,
                        'better_before': name_j if diff[k] < 0 else name_i,
                        'better_after': better_after
                    })
            
            crossover_points[f"{name_i}_vs_{name_j}"] = crossovers
        
        return crossover_points
    
    def _calculate_efficiency_metrics(self, results):
        """Calculate efficiency metrics for models."""
        metrics = {}
        
        for name in self.model_names:
            # Data efficiency (accuracy gain per data point)
            acc_values = results[f'{name}_accuracy_mean']
            train_sizes = results['train_sizes']
            
            if len(train_sizes) > 1:
                # Absolute acc gain / (additional data points)
                data_efficiency = (acc_values[-1] - acc_values[0]) / (train_sizes[-1] - train_sizes[0])
            else:
                data_efficiency = 0
            
            # Time efficiency (accuracy gain per second of training)
            time_values = results[f'{name}_training_time_mean']
            if np.sum(time_values) > 0:
                time_efficiency = (acc_values[-1] - acc_values[0]) / np.sum(time_values)
            else:
                time_efficiency = 0
            
            # Convergence rate (how quickly model reaches 95% of its final accuracy)
            final_acc = acc_values[-1]
            target_acc = 0.95 * final_acc
            
            # Find the first index where accuracy exceeds the target
            convergence_idx = np.where(acc_values >= target_acc)[0]
            if len(convergence_idx) > 0:
                convergence_idx = convergence_idx[0]
                convergence_fraction = results['data_fractions'][convergence_idx]
                convergence_size = train_sizes[convergence_idx]
            else:
                convergence_fraction = 1.0
                convergence_size = train_sizes[-1]
            
            # Asymptotic performance (final accuracy)
            asymptotic_performance = final_acc
            
            # Collect metrics
            metrics[name] = {
                'data_efficiency': data_efficiency,
                'time_efficiency': time_efficiency,
                'convergence_fraction': convergence_fraction,
                'convergence_size': convergence_size,
                'asymptotic_performance': asymptotic_performance
            }
        
        return metrics
    
    def plot_accuracy_progression(self, figsize=(12, 8), save_path=None):
        """
        Plot accuracy progression as training data increases.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.results:
            raise ValueError("No results available. Run compare() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for name in self.model_names:
            mean = self.results[f'{name}_accuracy_mean']
            std = self.results[f'{name}_accuracy_std']
            ax.plot(self.results['data_fractions'], mean, marker='o', label=name)
            ax.fill_between(
                self.results['data_fractions'],
                mean - std,
                mean + std,
                alpha=0.2
            )
        
        # Mark crossover points
        for pair, crossovers in self.results['crossover_points'].items():
            for crossover in crossovers:
                ax.axvline(x=crossover['data_fraction'], linestyle='--', alpha=0.5, color='gray')
                ax.text(
                    crossover['data_fraction'], 
                    ax.get_ylim()[0] + 0.02, 
                    f"Crossover: {crossover['data_fraction']:.2f}",
                    rotation=90, 
                    verticalalignment='bottom'
                )
        
        ax.set_xlabel('Training Data Fraction')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Progression with Increasing Training Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_time(self, figsize=(12, 8), save_path=None):
        """
        Plot training time as training data increases.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.results:
            raise ValueError("No results available. Run compare() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for name in self.model_names:
            mean = self.results[f'{name}_training_time_mean']
            std = self.results[f'{name}_training_time_std']
            ax.plot(self.results['data_fractions'], mean, marker='o', label=name)
            ax.fill_between(
                self.results['data_fractions'],
                mean - std,
                mean + std,
                alpha=0.2
            )
        
        ax.set_xlabel('Training Data Fraction')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Training Time with Increasing Training Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_efficiency_metrics(self, figsize=(14, 8), save_path=None):
        """
        Plot efficiency metrics comparison.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.results:
            raise ValueError("No results available. Run compare() first.")
        
        metrics = self.results['efficiency_metrics']
        
        # Prepare data for plotting
        model_names = list(metrics.keys())
        data_efficiency = [metrics[name]['data_efficiency'] for name in model_names]
        time_efficiency = [metrics[name]['time_efficiency'] for name in model_names]
        convergence_fraction = [metrics[name]['convergence_fraction'] for name in model_names]
        asymptotic_performance = [metrics[name]['asymptotic_performance'] for name in model_names]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Data efficiency
        axes[0, 0].bar(model_names, data_efficiency)
        axes[0, 0].set_title('Data Efficiency')
        axes[0, 0].set_ylabel('Accuracy Gain per Data Point')
        
        # Time efficiency
        axes[0, 1].bar(model_names, time_efficiency)
        axes[0, 1].set_title('Time Efficiency')
        axes[0, 1].set_ylabel('Accuracy Gain per Second')
        
        # Convergence rate
        axes[1, 0].bar(model_names, convergence_fraction)
        axes[1, 0].set_title('Convergence Rate')
        axes[1, 0].set_ylabel('Data Fraction to Reach 95% Accuracy')
        
        # Asymptotic performance
        axes[1, 1].bar(model_names, asymptotic_performance)
        axes[1, 1].set_title('Asymptotic Performance')
        axes[1, 1].set_ylabel('Final Accuracy')
        
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_convergence_visualization(self, figsize=(12, 8), save_path=None):
        """
        Plot visualization of convergence rates.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.results:
            raise ValueError("No results available. Run compare() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        max_acc = 0
        for name in self.model_names:
            mean = self.results[f'{name}_accuracy_mean']
            max_acc = max(max_acc, np.max(mean))
            
            # Normalize accuracy to final value
            normalized = mean / mean[-1]
            ax.plot(self.results['data_fractions'], normalized, marker='o', label=name)
            
            # Mark 95% point
            idx = np.where(normalized >= 0.95)[0]
            if len(idx) > 0:
                idx = idx[0]
                ax.scatter(
                    self.results['data_fractions'][idx],
                    normalized[idx],
                    s=100,
                    marker='*',
                    color='red',
                    zorder=10
                )
                ax.text(
                    self.results['data_fractions'][idx],
                    normalized[idx] - 0.05,
                    f"{name}: {self.results['data_fractions'][idx]:.2f}",
                    horizontalalignment='center'
                )
        
        ax.axhline(y=0.95, linestyle='--', color='gray', alpha=0.5)
        ax.text(0.01, 0.95, '95% of final accuracy', verticalalignment='bottom')
        
        ax.set_xlabel('Training Data Fraction')
        ax.set_ylabel('Normalized Accuracy (Relative to Final)')
        ax.set_title('Convergence Rate Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def summarize_results(self):
        """
        Summarize results of the comparison.
        
        Returns:
        --------
        str
            Text summary of the results
        """
        if not self.results:
            return "No results available. Run compare() first."
        
        summary = []
        summary.append("=== Model Comparison Summary ===\n")
        
        # Efficiency metrics
        summary.append("Efficiency Metrics:")
        metrics = self.results['efficiency_metrics']
        for name in self.model_names:
            summary.append(f"\n{name}:")
            summary.append(f"  - Data Efficiency: {metrics[name]['data_efficiency']:.6f} (accuracy gain per data point)")
            summary.append(f"  - Time Efficiency: {metrics[name]['time_efficiency']:.6f} (accuracy gain per second)")
            summary.append(f"  - Convergence at: {metrics[name]['convergence_fraction']:.2f} data fraction ({metrics[name]['convergence_size']} samples)")
            summary.append(f"  - Final Accuracy: {metrics[name]['asymptotic_performance']:.4f}")
        
        # Crossover points
        summary.append("\nCrossover Points:")
        for pair, crossovers in self.results['crossover_points'].items():
            if crossovers:
                for i, crossover in enumerate(crossovers):
                    summary.append(f"\n{pair} - Crossover {i+1}:")
                    summary.append(f"  - Data Fraction: {crossover['data_fraction']:.2f}")
                    summary.append(f"  - Approx. Training Size: {crossover['train_size_approx']}")
                    summary.append(f"  - Better Before: {crossover['better_before']}")
                    summary.append(f"  - Better After: {crossover['better_after']}")
            else:
                summary.append(f"\n{pair}: No crossover detected")
        
        # Final performance comparison
        summary.append("\nFinal Performance Comparison:")
        final_accs = [(name, self.results[f'{name}_accuracy_mean'][-1]) for name in self.model_names]
        final_accs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, acc) in enumerate(final_accs):
            summary.append(f"  {i+1}. {name}: {acc:.4f}")
        
        # Training time comparison
        summary.append("\nFinal Training Time Comparison:")
        final_times = [(name, self.results[f'{name}_training_time_mean'][-1]) for name in self.model_names]
        final_times.sort(key=lambda x: x[1])
        
        for i, (name, time_value) in enumerate(final_times):
            summary.append(f"  {i+1}. {name}: {time_value:.4f} seconds")
        
        return "\n".join(summary)

def demo():
    """Demonstrate the usage of the PCA vs Neural Network comparison framework."""
    # Generate synthetic data
    from sklearn.datasets import make_classification
    
    # Parameters
    n_samples = 1000
    n_features = 20
    n_classes = 2
    
    # Generate data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=10,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create PCA classifier
    pca_model = PCAClassifier(n_components=10)
    
    # Create neural network
    nn_model = NeuralNetwork(
        input_size=n_features,
        hidden_layers=[20, 10],
        output_size=1,  # Binary classification
        activation='relu',
        output_activation='sigmoid'
    )
    
    # Create model comparison
    models = [pca_model, nn_model]
    model_names = ['PCA + LogReg', 'Neural Network']
    comparison = ModelComparison(models, model_names)
    
    # Compare models
    data_fractions = np.linspace(0.1, 1.0, 10)
    results = comparison.compare(X_train_scaled, y_train, test_size=0.2, data_fractions=data_fractions, n_repeats=3)
    
    # Print summary
    print(comparison.summarize_results())
    
    # Plot results
    comparison.plot_accuracy_progression()
    comparison.plot_training_time()
    comparison.plot_efficiency_metrics()
    comparison.plot_convergence_visualization()
    
    plt.show()

if __name__ == "__main__":
    demo()