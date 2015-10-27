from __future__ import print_function
import caffe
import numpy as np
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import h5py
import Image
from game_globals import *


class FeatureNet(object):

    def __init__(self, proto_filename=AUTOENCODER_TEST_PROTO, model_filename=AUTOENCODER_MODEL):
        #caffe.set_mode_cpu()
        self.net = caffe.Net(proto_filename, model_filename, caffe.TEST)

    def encodeHDF5(self, hdf5filename, index):
        h5 = h5py.File(hdf5filename, 'r')
        dataset = h5['data']
        my_image = np.array(dataset[index], dtype=np.float32)
        input_array = my_image[np.newaxis, np.newaxis, :, :]
        return self.encodeNumpyArray(input_array)
   
    def encodeImage(self, image_filename):
        my_image = np.array(Image.open(image_filename), dtype=np.float32)
        # Scale the image
        my_image *= (1/255.)
        input_array = my_image[np.newaxis, np.newaxis, :, :]
        return self.encodeNumpyArray(input_array)
    
    def encodeNumpyArray(self, arr):
        #print(arr)
        self.net.blobs['data'].data[...] = arr
        out = self.net.forward()

        # encode4 is the name of the model layer at the top of the autoencoder
        return self.net.blobs['encode4'].data[...]


if __name__=="__main__":
    fn = FeatureNet()
    #print(fn.encodeImage("../screenshots/test2.png"))
    print(fn.encodeHDF5("datasets/dataset.hdf5", 56423))
    
