import sys
sys.path.append('/ghome/minsb/tools/caffe_ada/python')
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
parser.add_argument('--weights', type=str, help='Path to the pre-trained vgg16 model', default='./pretrained_models/vgg16/vgg16_imagenet.caffemodel')
parser.add_argument('--gpu', type=int, help='ID of the GPU to use', default=0)
args = parser.parse_args()

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 

# Some other parameters, usually you don't need to change this
init_weights = os.path.abspath(args.weights)

# pycaffe
caffe.set_device(args.gpu)
caffe.set_mode_gpu()

net = caffe.Net('adapooling.prototxt', init_weights, caffe.TEST)

n_images = 5794
p = np.zeros([n_images])
att = np.zeros([n_images,14,14])
labels = np.zeros([n_images])
for it in range(0,n_images,1):
    net.forward()
    p[it] = net.blobs['p/relu'].data[0]
    att[it,...] = net.blobs['s/reshape2'].data[0][0]

    labels[it] = net.blobs['label'].data[0]

''' Read labels '''
fid = open('../data/val_imagelist.txt')
imnames = []
for line in fid.readlines():
  imnames.append(line.strip().split(' ')[0])

# write
import xlwt
workbook = xlwt.Workbook()
worksheet = workbook.add_sheet('sheet1')
for i in range(n_images):
  worksheet.write(i,0, imnames[i])
  worksheet.write(i,1, np.round(p[i],3)+1)
  worksheet.write(i,2, labels[i])
workbook.save('cub_p.xls')

import scipy
for i in range(n_images):
  att_ = np.squeeze(att[i,...])
  scipy.misc.imsave('/gdata/minsb/temp/'+str(i)+'.jpg', att_)


