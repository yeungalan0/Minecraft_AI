import numpy as np
import random
from Experience import Experience

class ReplayMemory:
    # Numpy array implementation
    def __init__(self):
        # Create an empty numpy array with 2000000 elements
        self.size = 2000000
        self.storage = np.empty((self.size,), dtype=object)
        self.index = 0
        
    def store(self, data_tuple):
        """Store the given object in the storage array"""
        # Continually replace the oldest element in the array
        if self.index >= self.size:
            self.index = 0
        self.storage[self.index] = data_tuple
        self.index += 1

    # Python list implementation
    # def __init__(self):
    #     # Create an empty numpy array with 2000000 elements
    #     self.size = 2000000
    #     self.storage = []
    #     self.index = 0
        
    # def store(self, data_tuple):
    #     """Store the given object in the storage array"""
    #     # Continually replace the oldest element in the array
    #     if len(self.storage) == self.size:
    #         if self.index >= self.size:
    #             self.index = 0
    #         self.storage[self.index] = data_tuple
    #         self.index += 1
    #     else:
    #         self.storage.append(data_tuple)

    def get_random(self, LEARNING_SAMPLE_SIZE):
        # TODO: Consider when this occurs and if the array is near empty...
        picked = []
        for x in range(LEARNING_SAMPLE_SIZE):
            picked.append(self.storage[random.randint(0, self.index)])
        return picked

    def print_storage(self, range_lower = None, range_upper = None):
        # For testing purposes
        if range_lower == None and range_upper == None:
            print(self.storage)
        else:
            print(self.storage[range_lower:range_upper])

if __name__ == "__main__":
    # Perform tests
    ds = ReplayMemory()
    for x in range(0,3):
        y = Experience(x, x, x, x)
        ds.store(y)
    ds.print_storage(0, 4)
    print(ds.get_random(32))

# Numpy array results, n = 30,000,000:
# real	0m13.407s
# user	0m13.078s
# sys	0m0.279s

# Python list results, n = 30,000,000:
# real	0m15.228s
# user	0m14.727s
# sys	0m0.499s

