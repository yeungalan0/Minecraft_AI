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
my_image = np.array(Image.open('screenshots/test.png'), dtype=np.float64)
input_array = combined_array[np.newaxis, np.newaxis, :, :]
input_array = np.array(input_array, dtype=np.float32)

lbs = np.array([1.0] * output_vector_length)
lbs_array = lbs[np.newaxis, np.newaxis, np.newaxis, :]
labels_array = np.array(lbs, dtype=np.float64)

print(input_array.shape)
print(labels_array.shape)

# print(input_array)
# print(labels_array)

# fake_lbs = np.array([1.0])
# fake_lbs_array = fake_lbs[:, np.newaxis, np.newaxis, np.newaxis]
# fake_labels_array = np.array(fake_lbs_array, dtype=np.float32)

fake_labels = np.array([[[[1]]]], dtype=np.float64)

# Set input arrays for both training and testing nets
#mysolver.net.blobs['data'].data[...] = input_array
#mysolver.net.blobs['label'].data[...] = labels_array

# mysolver.test_nets[0].set_input_arrays(input_array, labels_array)
# mysolver.test_nets[0].blobs['data'].data[...] = input_array
# mysolver.test_nets[0].blobs['label'].data[...] = labels_array

mysolver.net.set_input_arrays(input_array, fake_labels)
mysolver.test_nets[0].set_input_arrays(input_array, fake_labels)

mysolver.net.blobs['data'].data[...] = input_array
mysolver.net.blobs['label'].data[...] = labels_array
mysolver.test_nets[0].blobs['data'].data[...] = input_array
mysolver.test_nets[0].blobs['label'].data[...] = labels_array


for i in range(100):
    mysolver.step(1)
    
    if i % 1 == 0:
        out = mysolver.net.blobs['final_output'].data[...]
        #out2 = mysolver.test_nets[0].forward()
        #out3 = mysolver.test_nets[0].forward()
        print(out)
        #print(out2)

