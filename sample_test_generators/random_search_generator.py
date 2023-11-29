from random import randint
from code_pipeline.tests_generation import RoadTestFactory
from time import  sleep

import logging as log

class RandomSearchTestGenerator():
    def __init__(self, executor=None, map_size=None, max_iterations = 100):
        self.executor = executor
        self.map_size = map_size
        self.max_iterations = max_iterations
        nest_oob_percantage = float('-inf')  # Initialize with negative infinity

    def start(self):
        iteration = 0

        while not self.executor.is_over() and iteration < self.max_iterations:
            # Some debugging
            time_remaining = self.executor.get_remaining_time()["time-budget"]
            log.info(f"Starting test generation. Remaining time {time_remaining}")

            # Simulate the time to generate a new test
            sleep(0.5)
            # Pick up random points from the map. They will be interpolated anyway to generate the road
            road_points = []
            for i in range(0, 3):
                road_points.append((randint(0, self.map_size), randint(0, self.map_size)))

            # Some more debugging
            log.info("Generated test using: %s", road_points)
            # Decorate the_test object with the id attribute
            the_test = RoadTestFactory.create_road_test(road_points)
            

            
            test_outcome, description, execution_data = self.executor.execute_test(the_test)
            max_oob_percentage = max([state.oob_percentage for state in execution_data])
            time_remaining = self.executor.get_remaining_time()["time-budget"]
            log.info(f"Executed test {the_test.id}. Remaining time {time_remaining}")

            # Check if current solution is better than the previous best
            if oob_percentage > self.best_oob_percentage:
                self.best_oob_percentage = oob_percentage
                self.best_road_points = road_points
            # Print the result from the test and continue
            log.info("test_outcome %s", test_outcome)
            log.info("description %s", description)
            
        log.info("Random search completed.")
        log.info("Best road points found: %s", self.best_road_points)
        log.info("Best Out-of-Bounds Percentage: %s", self.best_oob_percentage)