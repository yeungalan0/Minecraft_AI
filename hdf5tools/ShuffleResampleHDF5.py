import sys
import random
import h5py

WINDOW_SIZE = 32

if __name__=="__main__":
    f = h5py.File(sys.argv[1], 'r')
    o = h5py.File(sys.argv[2], 'w')
    sample_size = int(sys.argv[3])

    d = f['data']
    d2 = o.create_dataset("data", (sample_size, WINDOW_SIZE, WINDOW_SIZE))

    index = 0
    for i in range(sample_size):
        r = random.randrange(len(d))
        d2[index] = d[r]
        index += 1
    
    o.close()
    f.close()


        


