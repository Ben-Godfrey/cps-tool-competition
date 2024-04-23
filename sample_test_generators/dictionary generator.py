# Importing validate_test function from test_evaluation in code_pipeline folder
from code_pipeline.tests_generation import RoadTestFactory
from code_pipeline.validation import TestValidator
import random

class DictionarySetup():
    def __init__(self):
        self.map_width = 200
        self.map_height = 200
        self.validation = TestValidator(200)


    def generate_valid_test(self):
        """Generate a valid test case."""
        # Generate 5 sets of coordinates
        test_case = []
        for i in range(0, 5):
            test_case.append((random.randint(0, self.map_width), random.randint(0, self.map_height)))

        return test_case

    def generate_100_valid_tests(self):
        """Generate 100 valid test cases."""
        valid_tests = []
        while len(valid_tests) < 100:
            try:
                test_case = self.generate_valid_test()
                the_test = RoadTestFactory.create_road_test(test_case)
                is_valid, _ = self.validation.validate_test(the_test)
            except:
                is_valid = False

            if is_valid:
                valid_tests.append(test_case)
                print("Valid case generated")
            else:
                print("Invalid")

        return valid_tests

    def read_test_cases(self,file_name):
        """Read test cases from a text file and convert them to an array."""
        test_cases = []

        with open(file_name, 'r') as file:
            for line in file:
                # Split the line by commas and remove parentheses to extract coordinates
                coordinates = line.strip().replace('(', '').replace(')', '').split(',')

                # Convert coordinates to tuples and append to test_cases list
                test_case = [(float(coordinates[i]), float(coordinates[i + 1])) for i in range(0, len(coordinates), 2)]
                test_cases.append(test_case)

        return test_cases




generator = DictionarySetup()

# Generate 100 valid test cases
#valid_tests = generator.generate_100_valid_tests()

#with open("dictionary-cases.txt", 'w') as file:
    #for test_case in valid_tests:
        #line = ','.join([f"({x},{y})" for x, y in test_case])
        #file.write(f"{line}\n")

# Print the first 10 test cases as an example
#for i, test_case in enumerate(valid_tests[:10]):
    #print(f"Test Case {i+1}: {test_case}")


file_name = "dictionary-cases.txt"
test_cases = generator.read_test_cases(file_name)

print(test_cases[0])