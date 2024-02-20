import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

# Load Iris dataset
iris = load_iris()
X_iris = iris.data  # Use all features for simplicity
y_iris = iris.target  # Choose a target variable, e.g., iris.target_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Create and fit the Kriging surrogate model
regression_model = PolynomialRegressionSurrogate(degree=2)
regression_model.fit(X_train, y_train)

# Predict on the test set
predictions = regression_model.predict(X_test)

# Evaluate the model (e.g., using mean squared error)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)