"""
Multi-Layer Perceptron Implementation in Python
A neural network with multiple hidden layers for binary classification.
"""

import numpy as np


# Activation Functions
def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(a):
    """Derivative of sigmoid given activation a = sigmoid(z)."""
    return a * (1 - a)


def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU."""
    return (z > 0).astype(float)


def tanh(z):
    """Tanh activation function."""
    return np.tanh(z)


def tanh_derivative(a):
    """Derivative of tanh given activation a = tanh(z)."""
    return 1 - a**2


def binary_cross_entropy(y_true, y_pred):
    """
    Binary cross-entropy loss.
    
    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted probabilities (n_samples,)
    
    Returns:
        Scalar loss value
    """
    epsilon = 1e-15  # Avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class Layer:
    """A single fully-connected layer."""
    
    def __init__(self, input_size, output_size, activation='sigmoid'):
        """
        Initialize layer parameters.
        
        Args:
            input_size: Number of input features
            output_size: Number of neurons in this layer
            activation: 'sigmoid', 'relu', or 'tanh'
        """
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (input_size + output_size))
        self.W = np.random.uniform(-limit, limit, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        
        # Activation function
        self.activation = activation
        if activation == 'sigmoid':
            self.activate = sigmoid
            self.activate_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activate = relu
            self.activate_derivative = relu_derivative
        elif activation == 'tanh':
            self.activate = tanh
            self.activate_derivative = tanh_derivative
        
        # Cache for backprop
        self.z = None
        self.a = None
        self.x = None
    
    def forward(self, x):
        """
        Forward pass through the layer.
        
        Args:
            x: Input of shape (input_size, batch_size)
        
        Returns:
            Activation of shape (output_size, batch_size)
        """
        self.x = x
        self.z = self.W @ x + self.b
        self.a = self.activate(self.z)
        return self.a
    
    def backward(self, delta_next):
        """
        Backward pass through the layer.
        
        Args:
            delta_next: Gradient from next layer (output_size, batch_size)
        
        Returns:
            delta: Gradient to pass to previous layer (input_size, batch_size)
            dW: Weight gradient (output_size, input_size)
            db: Bias gradient (output_size, 1)
        """
        batch_size = self.x.shape[1]
        
        # Gradient w.r.t. pre-activation
        if self.activation == 'sigmoid' or self.activation == 'tanh':
            delta = delta_next * self.activate_derivative(self.a)
        else:  # ReLU
            delta = delta_next * self.activate_derivative(self.z)
        
        # Gradients for parameters
        dW = delta @ self.x.T / batch_size
        db = np.sum(delta, axis=1, keepdims=True) / batch_size
        
        # Gradient to pass to previous layer
        delta_prev = self.W.T @ delta
        
        return delta_prev, dW, db


class MLP:
    """Multi-Layer Perceptron for binary classification."""
    
    def __init__(self, layer_sizes, activations=None):
        """
        Initialize MLP.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activations: List of activation functions for each layer (default: all sigmoid)
        """
        self.layers = []
        
        if activations is None:
            activations = ['sigmoid'] * (len(layer_sizes) - 1)
        
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)
        
        self.losses = []
    
    def forward(self, X):
        """
        Forward pass through all layers.
        
        Args:
            X: Input of shape (input_size, batch_size)
        
        Returns:
            Output of shape (output_size, batch_size)
        """
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a
    
    def backward(self, y_true, y_pred):
        """
        Backward pass through all layers.
        
        Args:
            y_true: True labels (output_size, batch_size)
            y_pred: Predicted values (output_size, batch_size)
        
        Returns:
            List of (dW, db) tuples for each layer
        """
        # Initial gradient (for binary cross-entropy + sigmoid)
        delta = y_pred - y_true
        
        gradients = []
        
        # Backpropagate through layers in reverse
        for i in range(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                # Output layer: delta already computed
                batch_size = self.layers[i].x.shape[1]
                dW = delta @ self.layers[i].x.T / batch_size
                db = np.sum(delta, axis=1, keepdims=True) / batch_size
                delta_prev = self.layers[i].W.T @ delta
            else:
                # Hidden layers
                delta_prev, dW, db = self.layers[i].backward(delta)
            
            gradients.insert(0, (dW, db))
            delta = delta_prev
        
        return gradients
    
    def update_parameters(self, gradients, learning_rate):
        """Update all layer parameters."""
        for layer, (dW, db) in zip(self.layers, gradients):
            layer.W -= learning_rate * dW
            layer.b -= learning_rate * db
    
    def fit(self, X, y, learning_rate=0.1, epochs=1000, verbose=True):
        """
        Train the MLP.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Labels of shape (n_samples,)
            learning_rate: Learning rate
            epochs: Number of training epochs
            verbose: Whether to print progress
        """
        # Transpose to (features, samples)
        X_T = X.T
        y_T = y.reshape(1, -1)
        
        self.losses = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X_T)
            
            # Compute loss
            loss = binary_cross_entropy(y_T.flatten(), y_pred.flatten())
            self.losses.append(loss)
            
            # Backward pass
            gradients = self.backward(y_T, y_pred)
            
            # Update parameters
            self.update_parameters(gradients, learning_rate)
            
            # Print progress
            if verbose and (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        if verbose:
            print(f"\nTraining completed. Final loss: {self.losses[-1]:.6f}")
    
    def predict_proba(self, X):
        """Predict probabilities."""
        X_T = X.T
        return self.forward(X_T).T
    
    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int).flatten()
    
    def score(self, X, y):
        """Compute accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


if __name__ == "__main__":
    # Example: XOR problem
    print("Training MLP on XOR problem\n")
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    # Create and train MLP
    mlp = MLP(layer_sizes=[2, 4, 1], activations=['tanh', 'sigmoid'])
    mlp.fit(X, y, learning_rate=0.5, epochs=5000, verbose=True)
    
    # Test predictions
    print("\n--- Predictions ---")
    for i in range(len(X)):
        pred_proba = mlp.predict_proba(X[i:i+1])[0, 0]
        pred_class = mlp.predict(X[i:i+1])[0]
        print(f"Input: {X[i]}, Predicted: {pred_class}, Probability: {pred_proba:.4f}, Actual: {y[i]}")
    
    print(f"\nAccuracy: {mlp.score(X, y):.2%}")
