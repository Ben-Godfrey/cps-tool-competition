import json

class TestResultParser():

    def __init__(self, test_cases_file, results_file):
        self.test_cases = self.read_test_cases(test_cases_file)
        self.results = self.read_results(results_file)

    def read_test_cases(self, file_name):
        """Read test cases from a text file and convert them to a list."""
        test_cases = []

        with open(file_name, 'r') as file:
            for line in file:
                # Split the line by commas and remove parentheses to extract coordinates
                coordinates = line.strip().replace('(', '').replace(')', '').split(',')

                # Convert coordinates to tuples and append to test_cases list
                test_case = [(float(coordinates[i]), float(coordinates[i + 1])) for i in range(0, len(coordinates), 2)]
                test_cases.append(test_case)

        return test_cases

    def read_results(self, file_name):
        """Read test results from a text file and convert them to a list."""
        results = []

        with open(file_name, 'r') as file:
            for line in file:
                # Convert the line to a float and append to results list
                results.append(float(line.strip()))

        return results

    def create_results_dictionary(self):
        """Create a dictionary where keys are test cases and values are max OOB percentages."""
        results_dict = {}

        for test_case, result in zip(self.test_cases, self.results):
            # Convert test_case to a string to use as key in dictionary
            key = str(test_case)

            # Use result as value in dictionary
            results_dict[key] = result

        return results_dict

    def write_results_to_file(self, results_dict, file_name):
        """Write results dictionary to a text file."""
        with open(file_name, 'w') as file:
            json.dump(results_dict, file, indent=4)

# File paths
test_cases_file = "dictionary-cases.txt"
results_file = "results.txt"
output_file = "results_dictionary.txt"

# Create TestResultParser instance
parser = TestResultParser(test_cases_file, results_file)

# Create results dictionary
results_dict = parser.create_results_dictionary()

# Write results dictionary to file
parser.write_results_to_file(results_dict, output_file)