import numpy as np
from scipy.spatial.distance import cdist

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
# Assume you have a set of input points X and corresponding output values y
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([3, 5, 7])

# Create and fit the Radial Basis Function Network
rbfn = RadialBasisFunctionNetwork(num_rbf_neurons=5, sigma=1.0)
rbfn.fit(X, y)

# Predict at new input points
new_points = np.array([[4, 5], [5, 6]])
predictions = rbfn.predict(new_points)

print("Predictions:", predictions)