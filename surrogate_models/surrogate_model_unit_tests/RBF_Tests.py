import unittest
import numpy as np
from surrogate_models.RBF import RadialBasisFunctionNetwork


class TestRBFNSurrogate(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 2)
        self.y_train = np.random.rand(100)

        # Initialize RBFN model
        self.model = RBF(num_rbf=10)

    def test_fit_predict(self):
        # Test fitting the model and making predictions
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_train)
        
        # Check that predictions have the correct shape
        self.assertEqual(y_pred.shape, self.y_train.shape)

    def test_fit_predict_with_custom_parameters(self):
        # Test fitting the model with custom parameters and making predictions
        centers = np.random.rand(10, 2)
        widths = np.random.rand(10)
        weights = np.random.rand(10)

        self.model = RBF(centers=centers, widths=widths, weights=weights)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_train)
        
        # Check that predictions have the correct shape
        self.assertEqual(y_pred.shape, self.y_train.shape)

    def test_fit_predict_with_single_data_point(self):
        # Test fitting the model and making predictions with a single data point
        X_single = np.random.rand(1, 2)
        y_single = np.random.rand(1)

        self.model.fit(X_single, y_single)
        y_pred = self.model.predict(X_single)
        
        # Check that predictions have the correct shape
        self.assertEqual(y_pred.shape, y_single.shape)

if __name__ == '__main__':
    unittest.main()