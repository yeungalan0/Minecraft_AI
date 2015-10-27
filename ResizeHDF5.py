import sys
import h5py

WINDOW_SIZE = 32

if __name__=="__main__":

    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    size = int(sys.argv[3])
    
    print(filename1, filename2, size)

    f = h5py.File(filename1, 'r')
    o = h5py.File(filename2, 'w')

    d = f['data']
    #d.resize((int(sys.argv[2]), WINDOW_SIZE, WINDOW_SIZE))

    d2 = o.create_dataset("data", (size, WINDOW_SIZE, WINDOW_SIZE))
    
    for i in range(size):
        d2[i] = d[i]

    f.close()
    o.close()



        


