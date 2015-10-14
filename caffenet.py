from __future__ import print_function
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

# helper function for common structures

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(lmdb, batch_size=32, include_acc=False):
    print ("building net")

    data, label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                         transform_param=dict(crop_size=84, mean_value=[104, 117, 123], mirror=True))
    
    # the net itself
    conv1, relu1 = conv_relu(data, 8, 32, stride=4)
    conv2, relu2 = conv_relu(relu1, 4, 16, stride=2)
    ip1 = L.InnerProduct(relu2, num_output=256)
    relu3 = L.ReLU(ip1, in_place=True)
    ip2 = L.InnerProduct(relu3, num_output=64)
    loss = L.SoftmaxWithLoss(ip2, label)

    if include_acc:
        acc = L.Accuracy(ip2, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)

def make_net():

    with open('train.prototxt', 'w') as f:
        f.write(str(caffenet('train_lmdb')))
        

    net = caffe.Net("train.prototxt", caffe.TRAIN)
    print (type(net))

#    with open('test.prototxt', 'w') as f:
#        print(caffenet('/path/to/caffe-val-lmdb', batch_size=50, include_acc=True), file=f)

if __name__ == '__main__':
    make_net()
