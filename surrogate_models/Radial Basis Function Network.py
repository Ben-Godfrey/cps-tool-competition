import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris

class RadialBasisFunctionNetwork:
    def __init__(self, num_rbf_neurons, sigma=1.0):
        self.num_rbf_neurons = num_rbf_neurons
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _radial_basis_function(self, x, c, sigma):
        # Radial basis function activation
        return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * sigma ** 2))

    def _compute_rbf_activations(self, X):
        # Compute radial basis function activations for all input samples
        rbf_activations = np.zeros((X.shape[0], self.num_rbf_neurons))
        for i in range(self.num_rbf_neurons):
            rbf_activations[:, i] = self._radial_basis_function(X, self.centers[i], self.sigma)
        return rbf_activations

    def fit(self, X, y):
        # Initialize RBF neuron centers using k-means clustering
        self.centers = X[np.random.choice(X.shape[0], self.num_rbf_neurons, replace=False)]

        # Compute RBF activations
        rbf_activations = self._compute_rbf_activations(X)

        # Solve for the weights using linear regression
        self.weights = np.linalg.pinv(rbf_activations) @ y

    def predict(self, X):
        # Compute RBF activations for new input samples
        rbf_activations = self._compute_rbf_activations(X)

        # Make predictions using the learned weights
        predictions = rbf_activations @ self.weights

        return predictions

# Example usage:
# Load Iris dataset
iris = load_iris()
X_iris = iris.data  # Use all features for simplicity
y_iris = iris.target  # Choose a target variable, e.g., iris.target_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Create and fit the Radial Basis Function Network surrogate model
rbf_model = RadialBasisFunctionNetwork(num_rbf_neurons=10, sigma=1.0)
rbf_model.fit(X_train, y_train)

# Predict on the test set
predictions = rbf_model.predict(X_test)

# Evaluate the model (e.g., using mean squared error)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)