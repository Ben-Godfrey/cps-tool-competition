import unittest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from surrogate_models.PolynomialRegression import PolynomialRegressionSurrogate

class TestPolynomialRegressionSurrogate(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data for testing
        np.random.seed(42)
        X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        
        # Instantiate PolynomialRegressionSurrogate with degree=2 for testing
        self.model = PolynomialRegressionSurrogate(degree=2)
        self.model.fit(self.X_train, self.y_train)

    def test_fit_predict(self):
        # Test fitting and predicting on training data
        y_pred_train = self.model.predict(self.X_train)
        self.assertEqual(y_pred_train.shape, self.y_train.shape)

    def test_predict(self):
        # Test predicting on test data
        y_pred_test = self.model.predict(self.X_test)
        self.assertEqual(y_pred_test.shape, self.y_test.shape)

    def test_mean_squared_error(self):
        # Test mean squared error on test data
        y_pred_test = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred_test)
        self.assertAlmostEqual(mse, 0.0, places=1)  # Assert that MSE is close to zero (since we generated noiseless data)

if __name__ == '__main__':
    unittest.main()