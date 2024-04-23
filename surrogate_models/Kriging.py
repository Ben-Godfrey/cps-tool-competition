import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class KrigingSurrogate:
    def __init__(self):
        # Set up the Gaussian Process Regressor with an RBF kernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=100)

    def fit(self, X, y):
        # Fit the surrogate model to the provided data
        self.gp.fit(X, y)

    def predict(self, X):
        # Predict the mean and standard deviation of the surrogate model at input points X
        mean, std = self.gp.predict(X, return_std=True)
        return mean, std

    def acquire_next_point(self, candidate_points):
        # Choose the next point to sample using an acquisition function (e.g., expected improvement)
        mean, std = self.predict(candidate_points)
        
        # Ensure std is not close to zero to avoid division by zero
        std = np.maximum(std, 1e-6)  # Set a minimum threshold for std
        
        # Calculate expected improvement
        expected_improvement = (mean - np.min(mean)) / std
        
        # If all std values are close to zero, set expected_improvement to zero
        if np.all(std < 1e-6):
            expected_improvement = np.zeros_like(expected_improvement)
        else:
            # Otherwise, proceed with normal calculation
            expected_improvement = (mean - np.min(mean)) / std
        
        # Find the index of the candidate point with maximum expected improvement
        next_point_index = np.argmax(expected_improvement)
        
        # Select the next point
        next_point = candidate_points[next_point_index]
        return next_point
# Example usage:

# Load Iris dataset
iris = load_iris()
X_iris = iris.data  # Use all features for simplicity
y_iris = iris.target  # Choose a target variable, e.g., iris.target_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Create and fit the Kriging surrogate model
kriging_model = KrigingSurrogate()
kriging_model.fit(X_train, y_train)

# Predict on the test set
predictions, std_devs = kriging_model.predict(X_test)

# Evaluate the model (e.g., using mean squared error)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)