from __future__ import print_function
import caffe
import numpy as np
import Image

from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2


caffe.set_mode_cpu()
mysolver = caffe.SGDSolver('minecraft_solver.prototxt')


for k, v in mysolver.net.blobs.items():
    print(k, v.data.shape)

# print(mysolver.net.blobs['label'].data[...])
# print("OUCH")

output_vector_length = 64

def setInput(mysolver, filename):
    my_image = np.array(Image.open('screenshots/%s.png' % filename), dtype=np.float32)
    my_image *= (1/255.)
    
    input_array = my_image.reshape((7056))
    labels = np.array([0.76] * output_vector_length)
    # print("INPUTARRAY1:", input_array)
    # print(labels)
    combined_array = np.append(input_array, labels)#input_array + labels #
    input_array = combined_array[np.newaxis, np.newaxis, np.newaxis, :]
    input_array = np.array(input_array, dtype=np.float32)
    print("INPUTARRAY2:", input_array)
    fake_labels = np.array([[[[1]]]], dtype=np.float32)
    
    # print("TOTALINPUTARRAY3BEFOREHAND:", mysolver.net.blobs['data'].data[...])
    mysolver.net.set_input_arrays(input_array, fake_labels)
    mysolver.test_nets[0].set_input_arrays(input_array, fake_labels)
    
    #print("TOTALINPUTARRAY3:", mysolver.net.blobs['data'].data[...])
    # print("INPUTARRAY3:", mysolver.net.blobs['input_data'].data[...])
    # print("LABELARRAY3:", mysolver.net.blobs['label_data'].data[...])


setInput(mysolver, "test3")
print(dir(mysolver))
print(dir(mysolver.net))
for i in range(1000):

    # if i % 2 == 0:
    #     setInput(mysolver, "test2")
    # else:
    #     setInput(mysolver, "test3")
        
    # mysolver.net.forward()
    # mysolver.net.backward()
    mysolver.step(1)
    
    if i % 1 == 0:
        data = mysolver.test_nets[0].blobs['data'].data[...]
        out = mysolver.net.blobs['final_output'].data[...]
        out2 = mysolver.net.forward()
        out3 = mysolver.test_nets[0].forward()
        out4 = mysolver.test_nets[0].blobs['final_output'].data[...]

        print("DATA:", data)
        # print("OUT:", out)
        #print("OUT4:", out4)

