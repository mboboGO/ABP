import sys
sys.path.append('../../apooling/caffe_ada/python')
import caffe
import h5py
import numpy as np
import os
import google.protobuf
import google.protobuf.text_format
import uuid
import pyprind
import argparse
import random



parser = argparse.ArgumentParser(description='Prepare fine-tuning of multiscale alpha pooling. The working directory should contain train_val.prototxt of vgg16. The models will be created in the subfolders.')
parser.add_argument('--init_weights', type=str, help='Path to the pre-trained vgg16 model', default='./pretrained_models/vgg16/vgg16_imagenet.caffemodel')
parser.add_argument('--gpu_id', type=int, help='ID of the GPU to use', default=0)
args = parser.parse_args()

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 

# Some other parameters, usually you don't need to change this
init_weights = os.path.abspath(args.init_weights)

# pycaffe
caffe.set_device(args.gpu_id)
caffe.set_mode_gpu()

solver = caffe.get_solver('adapooling.solver')
net = solver.net
solver.net.copy_from(init_weights)
for it in range(0,10000):
    solver.step(1)
    #net.forward()
    
    #solver.step(1)
    p = net.blobs['p/relu'].data[...]
    s = net.blobs['s/sigmoid'].data[0][0]
    s_diff = net.blobs['s/sigmoid'].diff[0][0]

    #print('p:',p)
    #print('s:',np.mean(s),np.min(s),np.max(s))
    #print('s:',s)
    #print('s_diff',s_diff)
