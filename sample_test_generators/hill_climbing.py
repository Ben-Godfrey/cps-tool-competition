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

class HillClimbingGenerator():

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
        self.restarts = 20

    def start(self):
        best_test = None
        best_oob_percentage = float('-inf')
        start_time = time.time()

        for i in range(self.restarts):
            current_test = self.generate_valid_test()
            current_oob_percentage = self.evaluate_test(current_test)

            iteration = 0
            while iteration < self.max_iterations / self.restarts:
                neighboring_tests = self.generate_neighbors(current_test)

                best_neighbor = None
                best_neighbor_oob_percentage = float('-inf')
                for neighbor in neighboring_tests:
                    the_test = RoadTestFactory.create_road_test(neighbor)
                    is_valid, _ = self.validation.validate_test(the_test)
                    if is_valid:
                        log.info("Valid neighbour!")
                        neighbor_oob_percentage = self.evaluate_test(neighbor)
                        if neighbor_oob_percentage > best_neighbor_oob_percentage:
                            best_neighbor = neighbor
                            best_neighbor_oob_percentage = neighbor_oob_percentage
                    else:
                        log.info("Neighbour was invalid")


                if best_neighbor_oob_percentage > current_oob_percentage:
                    current_test = best_neighbor
                    current_oob_percentage = best_neighbor_oob_percentage
                    log.info("Iteration %d, Improved OOB Percentage: %.3f", iteration + 1, current_oob_percentage)
                else:
                    log.info("Iteration %d, No Improvement", iteration + 1)
                    break

                iteration += 1

            if current_oob_percentage > best_oob_percentage:
                best_test = current_test
                best_oob_percentage = current_oob_percentage

            # Store the test results in the test_results dictionary
            self.test_results[str(best_test)] = best_oob_percentage

        # Save the test results to a file
        self.save_results()

        end_time = time.time()
        in_seconds = end_time - start_time
        elaspsed_time = str(datetime.timedelta(seconds=in_seconds))

        log.info("Hill climbing with random restarts completed.")
        log.info("Best test found: %s", best_test)
        log.info("Best Out-of-Bounds Percentage: %.3f", best_oob_percentage)
        log.info("Number of Invalid Initial Tests Generated: " + str(self.invalid))
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

    def generate_neighbors(self, test):
        """Generate neighboring tests by tweaking a single road point."""
        neighbors = []

        for i in range(len(test)):
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    new_x = test[i][0] + dx
                    new_y = test[i][1] + dy

                    # Ensure new coordinates are within map bounds
                    if 0 <= new_x < self.map_size and 0 <= new_y < self.map_size:
                        neighbor = test.copy()
                        neighbor[i] = (new_x, new_y)
                        neighbors.append(neighbor)

        return neighbors

    def save_results(self):
        """Save the test results to a text file."""
        print("Saving results")

        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
        file_path = os.path.join(script_dir, 'hill_climbing_results.txt')  # Create the file path

        with open(file_path, 'w') as file:
            for key, value in self.test_results.items():
                file.write(f"{key}: {value}\n")

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
