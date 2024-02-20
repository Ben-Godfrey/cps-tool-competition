import unittest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from surrogate_models.Kriging import KrigingSurrogate

class TestKrigingSurrogate(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data for testing
        np.random.seed(42)
        X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        
        # Instantiate KrigingSurrogate
        self.model = KrigingSurrogate()

    def test_fit_predict(self):
        # Fit the model to the training data
        self.model.fit(self.X_train, self.y_train)

        # Predict mean and standard deviation on the test data
        mean, std = self.model.predict(self.X_test)

        # Check that shapes of mean and std predictions match the input
        self.assertEqual(mean.shape, self.y_test.shape)
        self.assertEqual(std.shape, self.y_test.shape)

    def test_acquire_next_point(self):
        # Get number of features in data 
        num_features = self.X_train.shape[1]
        # Create some candidate points
        candidate_points = np.random.rand(10, num_features)

        # Acquire the next point
        next_point = self.model.acquire_next_point(candidate_points)

        # Ensure the next point is within the range of candidate points
        self.assertTrue(np.all(next_point >= 0) and np.all(next_point <= 1))
    
    def test_mean_squared_error(self):
        # Test mean squared error on test data
        y_pred_test = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred_test)
        self.assertAlmostEqual(mse, 0.0, places=1)  # Assert that MSE is close to zero (since we generated noiseless data)

if __name__ == '__main__':
    unittest.main()