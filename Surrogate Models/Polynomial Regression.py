import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class PolynomialRegressionSurrogate:
    def __init__(self, degree=2):
        # Set up Polynomial Regression with the specified degree
        self.poly_features = PolynomialFeatures(degree=degree)
        self.poly_regression = LinearRegression()

    def fit(self, X, y):
        # Transform the input features into polynomial features
        X_poly = self.poly_features.fit_transform(X)

        # Fit the Polynomial Regression model to the transformed data
        self.poly_regression.fit(X_poly, y)

    def predict(self, X):
        # Transform the input features into polynomial features
        X_poly = self.poly_features.transform(X)

        # Predict the output values using the Polynomial Regression model
        predictions = self.poly_regression.predict(X_poly)
        return predictions

# Example usage:
# Assume you have a set of input points X and corresponding output values y
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([3, 5, 7])

# Create and fit the Polynomial Regression surrogate model
poly_regression_model = PolynomialRegressionSurrogate(degree=2)
poly_regression_model.fit(X, y)

# Predict at new input points
new_points = np.array([[4, 5], [5, 6]])
predictions = poly_regression_model.predict(new_points)

print("Predictions:", predictions)