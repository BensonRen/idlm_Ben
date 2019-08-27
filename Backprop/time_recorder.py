import time
import numpy as np
"""
This is a class for keeping time and possible recording it to a file
"""

class time_keeper(object):
    def __init__(self, time_keeping_file = "time_keeper.txt"):
        self.start = time.time()
        self.time_keeping_file = time_keeping_file
        self.end = -1
        self.duration = -1
    def record(self, write_number):
        with open(self.time_keeping_file,"a") as f:
            self.end = time.time()
            self.duration = self.end - self.start
            f.write('{},{}\n'.format(write_number, self.duration))
