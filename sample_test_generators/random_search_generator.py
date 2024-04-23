from random import randint
from code_pipeline.tests_generation import RoadTestFactory
from time import sleep
import numpy as np
import logging as log
from surrogate_models.GeneralSurrogate import SurrogateModel
from code_pipeline.validation import TestValidator
import os
import random
import time
import datetime


class RandomSearchTestGenerator():
    def __init__(self, executor, map_size=None, max_iterations=100):
        self.surrogate_model = SurrogateModel("results_dictionary.txt", "kriging")
        self.map_size = map_size
        self.max_iterations = max_iterations
        self.best_predicted_value = float('-inf')  # Initialize with negative infinity
        self.best_road_points = None
        self.executor = executor
        self.validation = TestValidator(200)
        self.test_results = {}  # Dictionary to store test points and predicted values
        self.invalid = 0

    def start(self):
        iteration = 0
        start_time = time.time()


        while iteration < self.max_iterations:

            road_points = self.generate_valid_test()

            # Evaluate the test using the surrogate model
            predicted_value = self.evaluate_test(road_points)

            # Update the best solution if the current predicted value is better
            if predicted_value > self.best_predicted_value:
                self.best_predicted_value = predicted_value
                self.best_road_points = road_points

            # Store the test results in the test_results dictionary
            self.test_results[str(road_points)] = predicted_value

            # Print the result and continue
            log.info("Iteration: %d, Road Points: %s, Predicted Value: %.4f", iteration + 1, road_points,
                     predicted_value)

            iteration += 1

        # Save the test results to a file
        self.save_results()

        end_time = time.time()
        in_seconds = end_time - start_time
        elaspsed_time = str(datetime.timedelta(seconds=in_seconds))

        log.info("Random search completed.")
        log.info("Best road points found: %s", self.best_road_points)
        log.info("Best Predicted Out-of-Bounds Percentage: %s", self.best_predicted_value)
        log.info("Number of Invalid Tests Generated: " + str(self.invalid))
        log.info("Random search took: " + elaspsed_time)

    def generate_test(self):
        """Generate a valid test case."""
        # Generate 5 sets of coordinates
        test_case = []
        for i in range(0, 5):
            test_case.append((random.randint(0, self.map_size), random.randint(0, self.map_size)))

        return test_case

    def generate_valid_test(self):
        is_valid = False
        while is_valid == False:
            try:
                test_case = self.generate_test()
                the_test = RoadTestFactory.create_road_test(test_case)
                is_valid, _ = self.validation.validate_test(the_test)
            except:
                is_valid = False

            if is_valid:
                print("Valid case generated")
                return test_case
            else:
                print("Invalid")
                self.invalid += 1

    def evaluate_test(self, road_points):
        """Evaluate the test using the surrogate model and return the predicted value."""
        # Convert road points to a feature vector
        feature_vector = [coord for coord_pair in road_points for coord in coord_pair]
        feature_vector = np.array([feature_vector])

        # Predict the outcome using the surrogate model
        if self.surrogate_model.model_type == "kriging":
            y_pred, _ = self.surrogate_model.model.predict(feature_vector)  # Unpack the tuple
        else:
            y_pred = self.surrogate_model.model.predict(feature_vector)

        # Return the predicted value
        return y_pred[0]

    def save_results(self):
        """Save the test results to a text file."""
        print("Saving results")

        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
        file_path = os.path.join(script_dir, 'random_search_results.txt')  # Create the file path

        with open(file_path, 'w') as file:
            for key, value in self.test_results.items():
                file.write(f"{key}: {value}\n")
