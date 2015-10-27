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

for i in range(100):
    mysolver.step(1)
    
    if i % 1 == 0:
        data = mysolver.net.blobs['data'].data[...]
        out = mysolver.net.blobs['ip2'].data[...]
        loss = mysolver.net.blobs['loss_output'].data[...]
        #out2 = mysolver.test_nets[0].forward()
        print("DATA:", data)

        print("OUT:", out)
        print("LOSS: ", loss)
        #print(out2)

