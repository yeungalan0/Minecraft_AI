from __future__ import print_function
import caffe
import numpy as np
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2



class MinecraftNet:


    def __init__(self):
        #self.train_net, self.test_net = self.make_nets()
        caffe.set_mode_cpu()
        self.solver = caffe.SGDSolver('minecraft_solver.prototxt')
        self.default_data_init()
        #print (self.solver.net.blobs['data'].shape[0])
        #print([(k, v.data.shape) for k, v in self.solver.net.blobs.items()])


    def default_data_init(self):
        inp = np.ones((32, 1, 84, 84), dtype=np.float32)
        lab = np.ones((32,), dtype=np.float32)
        self.set_input_data(inp, lab)
        
        
    def load_model(self, path_to_model):
        #print ("loading model: ", path_to_model)
        self.solver.net.copy_from(path_to_model)
        

    def train(self, itrs):
        self.solver.step(itrs)


    def forward(self, data):
        data = np.array(data)
        print ("data1: ", data.shape)
        data = data.reshape((84, 84))
        print ("data2: ", data.shape)
        data_input = data[np.newaxis, np.newaxis, :]
        print ("data input shape", data_input.shape)
        self.solver.test_nets[0].blobs['data'].data[...] = data_input
        # out = self.solver.net.forward()
        out = self.solver.test_nets[0].forward()
        return out
        #output[it] = solver.test_nets[0].blobs['ip2'].data[:8]


    def set_input_data(self, i, l):
        self.solver.net.set_input_arrays(i, l)
        
        
    # helper function for common structures
    def conv_relu(self, bottom, ks, nout, stride=1, pad=0, group=1):
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                    num_output=nout, pad=pad, group=group)
        return conv, L.ReLU(conv, in_place=True)


    def ip_relu(self, bottom, nout):
        ip = L.InnerProduct(bottom, num_output=nout)
        return ip, L.ReLU(ip, in_place=True)


    def max_pool(self, bottom, ks, stride=1):
        return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


    def caffenet(self, lmdb, batch_size=32, include_acc=False):
        print ("building net")

        data, label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntops=2,
                             transform_param=dict(crop_size=84, mirror=True))
        
        # the net itself
        conv1, relu1 = self.conv_relu(data, 8, 32, stride=4)
        conv2, relu2 = self.conv_relu(relu1, 4, 16, stride=2)
        ip, relu3 = self.ip_relu(relu2, 256)        
        ip2 = L.InnerProduct(relu3, num_output=64)
        loss = L.SoftmaxWithLoss(ip2, label)

        if include_acc:
            acc = L.Accuracy(ip2, label)
            return to_proto(loss, acc)
        else:
            return to_proto(loss)


    def make_nets(self):
        with open('train.prototxt', 'w') as f:
            f.write(str(self.caffenet('train_lmdb')))
            
        train_net = caffe.Net("train.prototxt", caffe.TRAIN)

        with open('test.prototxt', 'w') as f:
            f.write(str(self.caffenet('test_lmdb', batch_size=32, include_acc=True)))

        test_net = caffe.Net("test.prototxt", caffe.TEST)

        return train_net, test_net

        

if __name__ == '__main__':
    mnet = MinecraftNet()
    #mnet.load_model('snapshots/minecraft/snapshots_iter_5.caffemodel')
    #mnet.train(5)
    
    
