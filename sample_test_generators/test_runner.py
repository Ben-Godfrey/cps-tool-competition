import numpy as np
import math
import logging as log
import matplotlib.pyplot as plt
import os
import csv
import json

from code_pipeline.tests_generation import RoadTestFactory


class TestRunner():

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

        # Get the directory of the current script file
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the relative path to the text file
        file_path = os.path.join(script_dir, "random_search_results.txt")
        self.dictionary = self.read_results_dict(file_path)

        self.testArray = self.read_test_cases(self.dictionary)

    def start(self):
        results = []
        for test in self.testArray:
            # Creating the RoadTest from the points
            print(test)
            the_test = RoadTestFactory.create_road_test(test)

            # Send the test for execution
            print("Executing test")
            test_outcome, description, execution_data = self.executor.execute_test(the_test)

            # Plot the OOB_Percentage: How much the car is outside the road?
            oob_percentage = [state.oob_percentage for state in execution_data]
            try:
                max_oob_percentage = max(oob_percentage)
            except:
                max_oob_percentage = None
            log.info("Collected %d states information. Max is %.3f", len(oob_percentage), max_oob_percentage)

            # Store the test case and max OOB percentage in results_dict
            results.append(max_oob_percentage)

            # Print test outcome
            log.info("test_outcome %s", test_outcome)
            log.info("description %s", description)

            import time
            time.sleep(10)

        # Write results_dict to a text file
        self.updateDictionary(results)
        self.save_results()

    def read_test_cases(self, file_name):
        """Convert each key of the results dictionary to a test case."""
        test_cases = []

        for key in self.dictionary.keys():
            # Remove leading and trailing brackets and split by '), ('
            cleaned_str = key.strip("[]")

            # Split the string by '), (' to get individual coordinate strings
            coordinate_strs = cleaned_str.split("), (")

            # Convert each coordinate string to a tuple of floats
            coordinates = [tuple(map(float, coord.strip("()").split(','))) for coord in coordinate_strs]

            test_cases.append(coordinates)

        return test_cases


    def read_results_dict(self, file_name):
        """Read results dictionary from a text file and convert it back to a dictionary."""
        results_dict = {}

        with open(file_name, 'r') as file:
            for line in file:
                test, predicted_value = line.strip().split(': ')
                results_dict[test] = [float(predicted_value), None]  # Placeholder for actual value

        return results_dict

    def save_results(self):
        """Save the test results to a text file."""
        print("Saving results")

        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
        file_path = os.path.join(script_dir, 'random_search_results.txt')  # Create the file path

        with open(file_path, 'w') as file:
            for key, (predicted_value, actual_value) in self.dictionary.items():
                file.write(f"{key}: {predicted_value}, {actual_value}\n")

    def updateDictionary(self, results):
        for (key, value), actual_value in zip(self.dictionary.items(), results):
            self.dictionary[key] = (value[0], actual_value)


