from __future__ import print_function
import caffe
import numpy as np
import Image

from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

batch_size = 1
number_outputs = 64

caffe.set_mode_cpu()

n = caffe.NetSpec()
n.data = caffe.layers.MemoryData(batch_size=batch_size, channels=1, height=84, width=84, ntop=1)
n.label = caffe.layers.MemoryData(batch_size=batch_size, channels=number_outputs, height=1, width=1, ntop=1)

# n.data, x = L.Data(batch_size=batch_size, channels=2, transform_param=dict(scale=1./255), ntop=2)
# y, n.label = L.Data(batch_size=batch_size, ntop=2)

n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
n.relu1 = L.ReLU(n.ip1, in_place=True)
n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
n.loss = L.EuclideanLoss(n.ip2, n.label)

o = open("test.prototxt", 'w')
o.write(str(n.to_proto()))
o.close()

net = caffe.Net('test.prototxt', caffe.TRAIN)

for k, v in net.blobs.items():
    print(k, v.data.shape)

output_vector_length = 64


my_image = np.array(Image.open('screenshots/test.png'))
input_array = my_image[np.newaxis, np.newaxis, :, :]
input_array = np.array(input_array, dtype=np.float32)

#lbs = np.array([1.0] * output_vector_length)
lbs = np.array([1.0] * 1)
#lbs_array = lbs[:, np.newaxis, np.newaxis, np.newaxis]
labels_array = np.array(lbs, dtype=np.float32)

# print(input_array)
# print(labels_array)

# fake_lbs = np.array([1.0])
# fake_lbs_array = fake_lbs[:, np.newaxis, np.newaxis, np.newaxis]
# fake_labels_array = np.array(fake_lbs_array, dtype=np.float32)

# Set input arrays for both training and testing nets
net.set_input_arrays(input_array, labels_array)
#mysolver.net.blobs['data'].data[...] = input_array
#mysolver.net.blobs['label'].data[...] = labels_array

# mysolver.test_nets[0].set_input_arrays(input_array, labels_array)
# mysolver.test_nets[0].blobs['data'].data[...] = input_array
# mysolver.test_nets[0].blobs['label'].data[...] = labels_array


for i in range(10):

    #mysolver.step(1)
    out = net.forward()
    #out2 = mysolver.test_nets[0].forward()
    #out3 = mysolver.test_nets[0].forward()
    print(out)
    #print(out2)



