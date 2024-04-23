import numpy as np
import json
import os

from surrogate_models.Kriging import KrigingSurrogate
from surrogate_models.PolynomialRegression import PolynomialRegressionSurrogate
from surrogate_models.RBF import RadialBasisFunctionNetwork

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class SurrogateModel:

    def __init__(self, data_file,model_type):
        self.X, self.y = self.load_data(data_file)
        self.model_type = model_type

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.model = self.train_model(model_type)


    def load_data(self, data_file):
        """Load data from the results dictionary and preprocess it."""
        with open(data_file, 'r') as file:
            data = json.load(file)

        X = []
        y = []
        for key, value in data.items():
            # Convert key (coordinates) to a feature vector
            feature_vector = [coord for coord_pair in eval(key) for coord in coord_pair]
            X.append(feature_vector)
            y.append(value)

        return np.array(X), np.array(y)

    def train_model(self,model_type):
        if model_type == "kriging":
            """Train the Kriging Surrogate model."""
            surrogate_model = KrigingSurrogate()
        elif model_type == "pr":
            surrogate_model = PolynomialRegressionSurrogate(degree=2)
        else:
            surrogate_model = RadialBasisFunctionNetwork(num_rbf_neurons=10)

        surrogate_model.fit(self.X_train, self.y_train)
        return surrogate_model

    def evaluate_model(self):
        """Evaluate the model on the test set."""
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print(f"Mean Squared Error: {mse}")



# File path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, '..', 'sample_test_generators', 'results_dictionary.txt')

#Model Type
model_type = "pr"

# Create SurrogateModel instance
surrogate_model = SurrogateModel(data_file,model_type)

#Evaluate the model
surrogate_model.evaluate_model()


