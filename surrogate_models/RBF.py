import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris

class RadialBasisFunctionNetwork:
    def __init__(self, num_rbf_neurons = 10, centers=None, widths=None, weights=None):
        self.num_rbf = num_rbf_neurons
        self.centers = centers
        self.widths = widths
        self.weights = weights

    def fit(self, X, y):
        # Initialize centers and widths if not provided
        if self.centers is None:
            self.centers = X[np.random.choice(X.shape[0], self.num_rbf, replace=False)]
        if self.widths is None:
            distances = cdist(self.centers, self.centers)
            self.widths = np.mean(distances, axis=1) / np.sqrt(2 * np.log(2))

        # Compute RBF values
        phi = np.exp(-cdist(X, self.centers) ** 2 / (2 * self.widths ** 2))

        # Solve for weights using pseudo-inverse
        self.weights = np.linalg.pinv(phi) @ y

    def predict(self, X):
        phi = np.exp(-cdist(X, self.centers) ** 2 / (2 * self.widths ** 2))
        return phi @ self.weights

# Example usage:
# Load Iris dataset
iris = load_iris()
X_iris = iris.data  # Use all features for simplicity
y_iris = iris.target  # Choose a target variable, e.g., iris.target_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Create and fit the Radial Basis Function Network surrogate model
rbf_model = RadialBasisFunctionNetwork(num_rbf_neurons=10)
rbf_model.fit(X_train, y_train)

# Predict on the test set
predictions = rbf_model.predict(X_test)

# Evaluate the model (e.g., using mean squared error)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)