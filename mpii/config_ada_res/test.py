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

def voc_ap(rec, prec):
  """
  ap = voc_ap(rec, prec)
  Computes the AP under the precision recall curve.
  """

  rec = rec.reshape(rec.size,1); prec = prec.reshape(prec.size,1)
  z = np.zeros((1,1)); o = np.ones((1,1));
  mrec = np.vstack((z, rec, o))
  mpre = np.vstack((z, prec, z))
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])

  I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
  ap = 0;
  for i in I:
    ap = ap + (mrec[i] - mrec[i-1])*mpre[i];
  return ap

def compute_precision_score_mapping(thresh, prec, score):
  ind = np.argsort(thresh);
  thresh = thresh[ind];
  prec = prec[ind];
  for i in xrange(1, len(prec)):
    prec[i] = max(prec[i], prec[i-1]);

  indexes = np.unique(thresh, return_index=True)[1]
  indexes = np.sort(indexes);
  thresh = thresh[indexes]
  prec = prec[indexes]

  thresh = np.vstack((min(-1000, min(thresh)-1), thresh[:, np.newaxis], max(1000, max(thresh)+1)));
  prec = np.vstack((prec[0], prec[:, np.newaxis], prec[-1]));

  f = interp1d(thresh[:,0], prec[:,0])
  val = f(score)
  return val

def calc_pr_ovr_noref(counts, out):
  """
  [P, R, score, ap] = calc_pr_ovr(counts, out, K)
  Input    :
    counts : number of occurrences of this word in the ith image
    out    : score for this image
    K      : number of references
  Output   :
    P, R   : precision and recall
    score  : score which corresponds to the particular precision and recall
    ap     : average precision
  """
  #binarize counts
  counts = np.array(counts > 0, dtype=np.float32);
  tog = np.hstack((counts[:,np.newaxis].astype(np.float64), out[:, np.newaxis].astype(np.float64)))
  ind = np.argsort(out)
  ind = ind[::-1]
  score = np.array([tog[i,1] for i in ind])
  sortcounts = np.array([tog[i,0] for i in ind])

  tp = sortcounts;
  fp = sortcounts.copy();
  for i in xrange(sortcounts.shape[0]):
    if sortcounts[i] >= 1:
      fp[i] = 0.;
    elif sortcounts[i] < 1:
      fp[i] = 1.;
  P = np.cumsum(tp)/(np.cumsum(tp) + np.cumsum(fp));

  numinst = np.sum(counts);

  R = np.cumsum(tp)/numinst

  ap = voc_ap(R,P)
  return P, R, score, ap

def compute_map(all_logits, all_labels):
  num_classes = all_logits.shape[1]
  APs = []
  for cid in range(num_classes):
    this_logits = all_logits[:, cid]
    this_labels = (all_labels == cid).astype('float32')
    if np.sum(this_labels) == 0:
      print('No positive videos for class {}. Ignoring...'.format(cid))
      continue
    _, _, _, ap = calc_pr_ovr_noref(this_labels, this_logits)
    APs.append(ap)
  mAP = np.mean(APs)
  return mAP, APs

# Some other parameters, usually you don't need to change this
init_weights = os.path.abspath(args.weights)

# pycaffe
caffe.set_device(args.gpu)
caffe.set_mode_gpu()

net = caffe.Net('adapooling.prototxt', init_weights, caffe.TEST)

n_images = 6988
p = np.zeros([n_images])
fc = np.zeros([n_images,393])
att = np.zeros([n_images,14,14])
labels = np.zeros([n_images])
for it in range(0,n_images,1):
    net.forward()
    p[it] = net.blobs['p/relu'].data[0]
    att[it,...] = net.blobs['s/reshape2'].data[0][0]

    fc[it,:] = softmax(net.blobs['fc8_ft'].data[0,:])
    labels[it] = net.blobs['label'].data[0]

''' Read labels '''
fid = open('../data/trainval_val.txt')
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
workbook.save('mpii_p.xls')

import scipy
for i in range(n_images):
  att_ = np.squeeze(att[i,...])
  scipy.misc.imsave('/gdata/minsb/temp/'+str(i)+'.jpg', att_)

'''Evaluation'''
pre = np.argmax(fc, axis=1)
labels = np.int32(labels)
acc = np.mean(pre == labels)
mAP,ap = compute_map(fc, labels)
print('acc:',acc)
print('map:',mAP)
