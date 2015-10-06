import numpy as np

class DataStorage:
    # Numpy array implementation
    # def __init__(self):
    #     # Create an empty numpy array with 2000000 elements
    #     # TODO: Change datatype
    #     self.size = 2000000
    #     self.storage = np.empty((self.size,), "i4")
    #     self.index = 0
        
    # def store(self, data_tuple):
    #     """Store the given object in the storage array"""
    #     # Continually replace the oldest element in the array
    #     if self.index >= self.size:
    #         self.index = 0
    #     self.storage[self.index] = data_tuple
    #     self.index += 1

    # Python list implementation
    def __init__(self):
        # Create an empty numpy array with 2000000 elements
        self.size = 2000000
        self.storage = []
        self.index = 0
        
    def store(self, data_tuple):
        """Store the given object in the storage array"""
        # Continually replace the oldest element in the array
        if len(self.storage) == self.size:
            if self.index >= self.size:
                self.index = 0
            self.storage[self.index] = data_tuple
            self.index += 1
        else:
            self.storage.append(data_tuple)



    def print_storage(self, range_lower = None, range_upper = None):
        # For testing purposes
        if range_lower == None and range_upper == None:
            print(self.storage)
        else:
            print(self.storage[range_lower:range_upper])

if __name__ == "__main__":
    # Perform tests
    ds = DataStorage()
    for x in range(0,20000001):
        ds.store(x)
    ds.print_storage(0, 4)

# Numpy array results:
# [20000000 18000001 18000002 18000003]

# real	0m9.650s
# user	0m9.437s
# sys	0m0.206s

# Python list results:
# [20000000, 18000001, 18000002, 18000003]

# real	0m10.555s
# user	0m10.144s
# sys	0m0.291s


