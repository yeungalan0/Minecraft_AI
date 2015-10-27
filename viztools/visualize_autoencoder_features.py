from __future__ import print_function
import caffe
import numpy as np
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import time
import h5py
import Image
import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
#plt.interactive(True)

WINDOW_SIZE = 32
TEST_PROTO = '../protos/autoencoder_tester.prototxt'
MODEL_FILE = '../models/autoencoder/current.caffemodel'
DATASET = '../datasets/dataset.hdf5'

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    print("SHOWING")
    print(data)
    print(data.shape)
    #plt.imshow(data)
    plt.imsave("grid.png", data)


caffe.set_mode_cpu()
net = caffe.Net(TEST_PROTO, MODEL_FILE, caffe.TEST)

for k, v in net.params.items():
    print(k, v[0].data.shape)


h5 = h5py.File(DATASET, 'r')
dataset = h5['data']

my_image = np.array(dataset[88], dtype=np.float32)
#my_image *= (1/255.)
print("INPUT:", my_image)

# my_image = my_image.reshape((7056))
input_array = my_image[np.newaxis, np.newaxis, :, :]


# my_image2 = np.array(Image.open('test2.png'), dtype=np.float32)
# my_image2 *= (1/255.)
# input_array2 = my_image2[np.newaxis, np.newaxis, :, :]

net.blobs['data'].data[...] = input_array
out = net.forward()
print("WITHINPUT:", net.blobs['final_data'].data)
#print("ENCODERWEIGHTS:", net.params['encoder'][0].data[...])
#print("DECODERWEIGHTS:", net.params['decoder'][0].data[...])

#print("TOPMOST FEATURES:")
#print(net.blobs['encode4'].data[...])



output_array = out['final_data']
output_array = output_array.reshape((WINDOW_SIZE,WINDOW_SIZE))

plt.imsave("output.png", output_array)
plt.imsave("input.png", my_image)


# filters = net.params['encode1'][0].data
# print(filters.shape)
# filters = filters.reshape((1000, 32, 32))
# vis_square(filters)

