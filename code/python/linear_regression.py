"""
Linear Regression from First Principles
Implements gradient descent-based linear regression with MSE loss.
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression Model using Gradient Descent.
    
    Model: y_hat = X @ w + b
    Loss: MSE = (1/n) * sum((y_hat - y)^2)
    
    Attributes:
        w (np.ndarray): Weight vector of shape (n_features,)
        b (float): Bias term
        losses (list): Training loss history
    """
    
    def __init__(self):
        """Initialize the linear regression model."""
        self.w = None
        self.b = None
        self.losses = []
    
    def forward(self, X):
        """
        Compute forward pass (predictions).
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
        
        Returns:
            np.ndarray: Predictions of shape (n_samples,)
        """
        return X @ self.w + self.b
    
    def compute_loss(self, y_hat, y):
        """
        Compute Mean Squared Error loss.
        
        Args:
            y_hat (np.ndarray): Predictions of shape (n_samples,)
            y (np.ndarray): True targets of shape (n_samples,)
        
        Returns:
            float: MSE loss
        """
        return np.mean((y_hat - y) ** 2)
    
    def compute_gradients(self, X, y, y_hat):
        """
        Compute gradients of loss w.r.t. w and b.
        
        Gradients:
            dw = (2/n) * X.T @ (y_hat - y)
            db = (2/n) * sum(y_hat - y)
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            y (np.ndarray): True targets of shape (n_samples,)
            y_hat (np.ndarray): Predictions of shape (n_samples,)
        
        Returns:
            tuple: (dw, db) gradients for weights and bias
        """
        n = X.shape[0]
        error = y_hat - y
        
        dw = (2.0 / n) * X.T @ error
        db = (2.0 / n) * np.sum(error)
        
        return dw, db
    
    def fit(self, X, y, learning_rate=0.01, epochs=1000, verbose=True):
        """
        Train the linear regression model using gradient descent.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Training targets of shape (n_samples,)
            learning_rate (float): Learning rate for gradient descent
            epochs (int): Number of training iterations
            verbose (bool): Whether to print training progress
        
        Returns:
            self: Returns the fitted model
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.w = np.random.randn(n_features)
        self.b = 0.0
        self.losses = []
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            y_hat = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y_hat, y)
            self.losses.append(loss)
            
            # Compute gradients
            dw, db = self.compute_gradients(X, y, y_hat)
            
            # Update parameters
            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db
            
            # Print progress
            if verbose and (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        if verbose:
            print(f"\nTraining completed. Final loss: {self.losses[-1]:.6f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
        
        Returns:
            np.ndarray: Predictions of shape (n_samples,)
        
        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if self.w is None or self.b is None:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        return self.forward(X)
    
    def get_parameters(self):
        """
        Get the learned model parameters.
        
        Returns:
            dict: Dictionary containing 'w' and 'b'
        """
        return {'w': self.w, 'b': self.b}
    
    def score(self, X, y):
        """
        Compute the R² score (coefficient of determination).
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            y (np.ndarray): True targets of shape (n_samples,)
        
        Returns:
            float: R² score
        """
        y_hat = self.predict(X)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def create_linear_regression_model():
    """
    Factory function to create a LinearRegression model instance.
    
    Returns:
        LinearRegression: A new linear regression model
    """
    return LinearRegression()


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Linear Regression from First Principles - Example")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 100
    n_features = 2
    
    X = np.random.randn(n_samples, n_features)
    true_w = np.array([3.0, 2.0])
    true_b = 1.0
    
    # Generate target with some noise
    y = X @ true_w + true_b + 0.5 * np.random.randn(n_samples)
    
    print(f"\nData shape: X={X.shape}, y={y.shape}")
    print(f"True parameters: w={true_w}, b={true_b:.4f}")
    
    # Create and train model
    print("\n" + "-" * 60)
    print("Training Model...")
    print("-" * 60)
    
    model = create_linear_regression_model()
    model.fit(X, y, learning_rate=0.1, epochs=1000)
    
    # Get learned parameters
    params = model.get_parameters()
    print(f"\nLearned parameters: w={params['w']}, b={params['b']:.4f}")
    print(f"True parameters: w={true_w}, b={true_b:.4f}")
    
    # Compute R² score
    r2 = model.score(X, y)
    print(f"\nR² Score: {r2:.6f}")
    
    # Make predictions on new data
    print("\n" + "-" * 60)
    print("Making Predictions on New Data...")
    print("-" * 60)
    
    X_new = np.array([[1.0, 2.0], [2.0, 3.0], [0.5, 1.5]])
    y_pred = model.predict(X_new)
    
    print(f"\nNew data:\n{X_new}")
    print(f"\nPredictions:\n{y_pred}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
