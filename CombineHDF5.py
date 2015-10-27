import sys
import h5py

WINDOW_SIZE = 32

if __name__=="__main__":
    f1 = h5py.File(sys.argv[1], 'r')
    f2 = h5py.File(sys.argv[2], 'r')
    o = h5py.File(sys.argv[3], 'w')

    d1 = f1['data']
    d2 = f2['data']

    combined_len = len(d1) + len(d2)

    d3 = o.create_dataset("data", (combined_len, WINDOW_SIZE, WINDOW_SIZE))
    
    index = 0
    for i in range(len(d1)):
        d3[index] = d1[i]
        index += 1
        
    for i in range(len(d2)):
        d3[index] = d2[i]
        index += 1
    
    o.close()



        


