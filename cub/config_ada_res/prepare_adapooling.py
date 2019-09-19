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

def calc_features(net, n_images, blobs):
    n_images = int(0.8*n_images)
    batchsize = net.blobs['data'].data.shape[0]
    feats = dict()
    for blob in blobs:
        out_shape = list(net.blobs[blob].data.shape)
        out_shape[0] = n_images
        print('Will allocate {:.2f} GiB of memory'.format(np.prod(out_shape)*2/1024/1024/1024))
        feats[blob] = np.zeros(tuple(out_shape),dtype=np.float16 if not blob=='label' else np.int32)
    print('Need %.3f GiB'%(np.sum([x.nbytes for x in feats.values()])/1024/1024/1024))

    for it in pyprind.prog_bar(range(0,n_images,batchsize),update_interval=10, stream=sys.stderr):
        net.forward()
        for blob in blobs:
            feats[blob][it:it+batchsize,...] = net.blobs[blob].data[:feats[blob][it:it+batchsize,...].shape[0],...]

    return [feats[blob] for blob in blobs]

parser = argparse.ArgumentParser(description='Prepare fine-tuning of multiscale alpha pooling. The working directory should contain train_val.prototxt of vgg16. The models will be created in the subfolders.')
parser.add_argument('train_imagelist', type=str, help='Path to imagelist containing the training images. Each line should contain the path to an image followed by a space and the class ID.')
parser.add_argument('val_imagelist', type=str, help='Path to imagelist containing the validation images. Each line should contain the path to an image followed by a space and the class ID.')
parser.add_argument('--init_weights', type=str, help='Path to the pre-trained vgg16 model', default='./pretrained_models/vgg16/vgg16_imagenet.caffemodel')
parser.add_argument('--gpu_id', type=int, help='ID of the GPU to use', default=0)
parser.add_argument('--save_path', type=str, help='saving path', default='/')
parser.add_argument('--num_classes', type=int, help='Number of object categories', default=1000)
parser.add_argument('--image_root', type=str, help='Image root folder, used to set the root_folder parameter of the ImageData layer of caffe.', default='/')
parser.add_argument('--architecture', type=str, help='CNN architecture to use as basis. Should be a folder name present in the ./pretrained_models/ directory. Should contain a prepared train_val.prototxt.', default='vgg16')
parser.add_argument('--chop_off_layer', type=str, help='Layer in the selected CNN architecture to compute the alpha pooling features from.', default='relu5_3')
parser.add_argument('--train_batch_size', type=int, help='Batch size in training. Should be between 1 and 8, as we will use iter_size to achieve an effective batch size of 8. For network with batch norm, a batch size of 4 or greater is required to avoid divergence and 8 is recommended if you have enough GPU memory.', default=8)
parser.add_argument('--resize', nargs='+', type=int, default=[480,480], help='The input size of the different multi-scale branches.')
parser.add_argument('--crop_size', type=int, default=448, help='The crop size of the augmented input image. Should be at least as high as the maximum of --resolutions' )
args = parser.parse_args()

# Some other parameters, usually you don't need to change this
chop_off_layer = args.chop_off_layer
resize_size = args.resize
crop_size = args.crop_size
num_classes = args.num_classes
init_weights = os.path.abspath(args.init_weights)

# pycaffe
caffe.set_device(args.gpu_id)
caffe.set_mode_gpu()

''' Building config for training '''
# Net
netparams_in = caffe.proto.caffe_pb2.NetParameter()
protofile = os.getcwd() + '/../pretrained_models/' + args.architecture +'/train_val.prototxt'
google.protobuf.text_format.Merge(open(protofile).read(),netparams_in)

assert(args.chop_off_layer in [l.name for l in netparams_in.layer]), 'Chop off layer not found. I can only find the layers {}'.format([l.name for l in netparams_in.layer])

''' Prepare data layer '''
lyr = netparams_in.layer
lyr[0].image_data_param.source = args.train_imagelist
lyr[0].image_data_param.root_folder = args.image_root
lyr[0].image_data_param.batch_size = args.train_batch_size
[lyr[0].image_data_param.smaller_side_size.append(0) for _ in range(2-len(lyr[0].image_data_param.smaller_side_size))]
lyr[0].type = 'ImageData'

lyr[1].image_data_param.source = args.val_imagelist
lyr[1].image_data_param.root_folder = args.image_root
lyr[1].image_data_param.batch_size = 1
lyr[1].type = 'ImageData'

lyr[0].transform_param.crop_size = crop_size
lyr[0].image_data_param.smaller_side_size[0] = resize_size[0]
lyr[0].image_data_param.smaller_side_size[1] = resize_size[1]

lyr[1].transform_param.crop_size = crop_size
[lyr[1].image_data_param.smaller_side_size.append(0) for _ in range(2-len(lyr[1].image_data_param.smaller_side_size))]
lyr[1].image_data_param.smaller_side_size[0] = resize_size[0]
lyr[1].image_data_param.smaller_side_size[1] = resize_size[0]
lyr[1].image_data_param.shuffle = False

# Add batch norm
netparams = caffe.proto.caffe_pb2.NetParameter()
netparams.name = netparams_in.name

alpha_outputs = []


''' backbone net'''
for idx, l in enumerate(netparams_in.layer):
    if l.type in ['ImageData', 'Data']:
        netparams.layer.add()
        netparams.layer[-1].MergeFrom(l)

    else:
        netparams.layer.add()
        netparams.layer[-1].MergeFrom(l)
        netparams.layer[-1].name = netparams.layer[-1].name 
        for param_idx, p in enumerate(netparams.layer[-1].param):
            p.name = '%s_param%i'%(l.name,param_idx)

        if l.name == chop_off_layer:
            last_conv = l.top[-1]
            break

netparams.layer.add()
netparams.layer[-1].name = 'top_bn'
netparams.layer[-1].type = 'BatchNorm'
netparams.layer[-1].bottom.append(last_conv)
netparams.layer[-1].top.append(netparams.layer[-1].name)
[netparams.layer[-1].param.add() for _ in range(3)]
netparams.layer[-1].param[0].lr_mult = 0.0
netparams.layer[-1].param[1].lr_mult = 0.0
netparams.layer[-1].param[2].lr_mult = 0.0

netparams.layer.add()
netparams.layer[-1].name = 'top_scale'
netparams.layer[-1].type = 'Scale'
netparams.layer[-1].bottom.append(netparams.layer[-2].name)
netparams.layer[-1].top.append(netparams.layer[-1].name)
[netparams.layer[-1].param.add() for _ in range(2)]
netparams.layer[-1].param[0].lr_mult = 1.0
netparams.layer[-1].param[0].decay_mult=1.0
netparams.layer[-1].param[1].lr_mult = 2.0
netparams.layer[-1].param[1].decay_mult=1.0
netparams.layer[-1].scale_param.bias_term = True

''' Add P-Net'''
if(1):
    netparams.layer.add()
    netparams.layer[-1].name = 'p/gpooling'
    netparams.layer[-1].type = 'Pooling'
    netparams.layer[-1].bottom.append('top_scale')
    netparams.layer[-1].top.append(netparams.layer[-1].name)
    netparams.layer[-1].pooling_param.pool = caffe.params.Pooling.AVE
    netparams.layer[-1].pooling_param.global_pooling = True
        
    netparams.layer.add()
    netparams.layer[-1].name = 'p/conv'
    netparams.layer[-1].type = 'Convolution'
    netparams.layer[-1].bottom.append(netparams.layer[-2].name)
    netparams.layer[-1].top.append(netparams.layer[-1].name)
    [netparams.layer[-1].param.add() for _ in range(2)]
    netparams.layer[-1].param[0].lr_mult = 0.0001
    netparams.layer[-1].param[0].decay_mult = 1.0
    netparams.layer[-1].param[1].lr_mult = 0.0002
    netparams.layer[-1].param[1].decay_mult = 0.0
    netparams.layer[-1].convolution_param.num_output=1
    netparams.layer[-1].convolution_param.kernel_size.append(1)
    netparams.layer[-1].convolution_param.weight_filler.type='gaussian'
    netparams.layer[-1].convolution_param.weight_filler.std=0.001
    netparams.layer[-1].convolution_param.bias_filler.type='constant'
    netparams.layer[-1].convolution_param.bias_filler.value=0.0
	
    netparams.layer.add()
    netparams.layer[-1].name = 'p/relu'
    netparams.layer[-1].type = 'ReLU'
    netparams.layer[-1].bottom.append(netparams.layer[-2].name)
    netparams.layer[-1].top.append(netparams.layer[-1].name)
	
    netparams.layer.add()
    netparams.layer[-1].name = 'p/att'
    netparams.layer[-1].type = 'Attention'
    netparams.layer[-1].bottom.append('top_scale')
    netparams.layer[-1].bottom.append(netparams.layer[-2].top[0])
    netparams.layer[-1].top.append(netparams.layer[-1].name)
    netparams.layer[-1].attention_param.operation=caffe.params.Attention.EXP

    p_feats = netparams.layer[-1].name
''' S-Net '''
if(1):
    netparams.layer.add()
    netparams.layer[-1].name = 's/conv'
    netparams.layer[-1].type = 'Convolution'
    netparams.layer[-1].bottom.append('top_scale')
    netparams.layer[-1].top.append(netparams.layer[-1].name)
    [netparams.layer[-1].param.add() for _ in range(2)]
    netparams.layer[-1].param[0].lr_mult = 0.0001
    netparams.layer[-1].param[0].decay_mult = 1.0
    netparams.layer[-1].param[1].lr_mult = 0.0002
    netparams.layer[-1].param[1].decay_mult = 0.0
    netparams.layer[-1].convolution_param.num_output=1
    netparams.layer[-1].convolution_param.kernel_size.append(1)
    netparams.layer[-1].convolution_param.weight_filler.type='gaussian'
    netparams.layer[-1].convolution_param.weight_filler.std=0.001
    netparams.layer[-1].convolution_param.bias_filler.type='constant'
    netparams.layer[-1].convolution_param.bias_filler.value=0.0

    netparams.layer.add()
    netparams.layer[-1].name = 's/reshape1'
    netparams.layer[-1].type = 'Reshape'
    netparams.layer[-1].bottom.append(netparams.layer[-2].name)
    netparams.layer[-1].top.append(netparams.layer[-1].name)
    netparams.layer[-1].reshape_param.shape.dim.append(0)
    netparams.layer[-1].reshape_param.shape.dim.append(0)
    netparams.layer[-1].reshape_param.shape.dim.append(1)
    netparams.layer[-1].reshape_param.shape.dim.append(-1)

    netparams.layer.add()
    netparams.layer[-1].name = 's/softmax'
    netparams.layer[-1].type = 'Softmax'
    netparams.layer[-1].bottom.append(netparams.layer[-2].name)
    netparams.layer[-1].top.append(netparams.layer[-1].name)
    netparams.layer[-1].softmax_param.axis = -1

    netparams.layer.add()
    netparams.layer[-1].name = 's/reshape2'
    netparams.layer[-1].type = 'Reshape'
    netparams.layer[-1].bottom.append(netparams.layer[-2].name)
    netparams.layer[-1].top.append(netparams.layer[-1].name)
    netparams.layer[-1].reshape_param.shape.dim.append(0)
    netparams.layer[-1].reshape_param.shape.dim.append(0)
    netparams.layer[-1].reshape_param.shape.dim.append(14)
    netparams.layer[-1].reshape_param.shape.dim.append(14)

    netparams.layer.add()
    netparams.layer[-1].name = 's/att'
    netparams.layer[-1].type = 'Attention'
    netparams.layer[-1].bottom.append('top_scale')
    netparams.layer[-1].bottom.append(netparams.layer[-2].top[0])
    netparams.layer[-1].top.append(netparams.layer[-1].name)
    netparams.layer[-1].attention_param.operation=caffe.params.Attention.SPACE

netparams.layer.add()
netparams.layer[-1].name = 'outer_product'
netparams.layer[-1].type = 'CompactBilinear'
netparams.layer[-1].bottom.append('p/att')
netparams.layer[-1].bottom.append('s/att')
netparams.layer[-1].top.append(netparams.layer[-1].name)
netparams.layer[-1].compact_bilinear_param.num_output = 8192

netparams.layer.add()
netparams.layer[-1].name = 'root'
netparams.layer[-1].type = 'SignedPower'
netparams.layer[-1].bottom.append(netparams.layer[-2].name)
netparams.layer[-1].top.append(netparams.layer[-1].name)
netparams.layer[-1].power_param.power = 0.5 
netparams.layer[-1].param.add()
netparams.layer[-1].param[0].lr_mult = 0
netparams.layer[-1].param[0].decay_mult = 0

netparams.layer.add()
netparams.layer[-1].name = 'l2'
netparams.layer[-1].type = 'L2Normalize'
netparams.layer[-1].bottom.append(netparams.layer[-2].top[0])
netparams.layer[-1].top.append(netparams.layer[-1].name)

# fc8
netparams.layer.add()
netparams.layer[-1].name = 'fc8_ft'
netparams.layer[-1].type = 'InnerProduct'
netparams.layer[-1].bottom.append(netparams.layer[-2].top[0])
netparams.layer[-1].top.append(netparams.layer[-1].name) 
netparams.layer[-1].inner_product_param.num_output = num_classes
[netparams.layer[-1].param.add() for _ in range(2)]
netparams.layer[-1].param[0].lr_mult = 1
netparams.layer[-1].param[0].decay_mult = 1
netparams.layer[-1].param[1].lr_mult = 2
netparams.layer[-1].param[1].decay_mult = 0

# Accuracy
netparams.layer.add()
netparams.layer[-1].name = 'loss'
netparams.layer[-1].type = 'SoftmaxWithLoss'
netparams.layer[-1].bottom.append(netparams.layer[-2].top[0])
netparams.layer[-1].bottom.append('label')
netparams.layer[-1].top.append(netparams.layer[-1].name)
netparams.layer[-1].loss_weight.append(1)

# Softmax
netparams.layer.add()
netparams.layer[-1].name = 'Accuracy'
netparams.layer[-1].type = 'Accuracy'
netparams.layer[-1].bottom.append(netparams.layer[-3].top[0])
netparams.layer[-1].bottom.append('label')
netparams.layer[-1].top.append(netparams.layer[-1].name) 
netparams.layer[-1].include.add()
netparams.layer[-1].include[0].phase = 1

for l in netparams.layer:
    if l.type == 'BatchNorm':
        l.batch_norm_param.moving_average_fraction = 0.997

print os.getcwd()
num_images = [len([None for _ in open(netparams.layer[i].image_data_param.source,'r')]) for i in [0,1]]
iter_per_epoch = int(num_images[0]/args.train_batch_size) 
assert iter_per_epoch>0

# Solver
solverfile = 'adapooling.solver'
params = caffe.proto.caffe_pb2.SolverParameter()
params.net = 'adapooling.prototxt'
params.test_iter.append(int(len([None for _ in open(netparams.layer[1].image_data_param.source,'rt')]) / lyr[1].image_data_param.batch_size))
params.test_interval = iter_per_epoch
params.test_initialization = True
params.base_lr = 0.001
params.display = 100
params.max_iter = 70 * iter_per_epoch
params.lr_policy = "step"
params.gamma = 0.5
params.stepsize = 10*iter_per_epoch
params.momentum = 0.9
params.weight_decay = 0.0005
params.snapshot = iter_per_epoch
params.snapshot_prefix = args.save_path
params.iter_size = int(8/lyr[0].image_data_param.batch_size)
assert params.iter_size > 0

open(solverfile,'w').write(google.protobuf.text_format.MessageToString(params))
open(params.net,'w').write(google.protobuf.text_format.MessageToString(netparams))

''' Extracting features '''

# train_feats
print('### Extracting training features...')
last_blob = [l.top[0] for l in netparams.layer if l.name == chop_off_layer][-1]
solver = caffe.get_solver('adapooling.solver')
solver.net.copy_from(init_weights)
train_feats,train_labels = calc_features(solver.net,num_images[0],[last_blob,'label'])
del solver

''' Building config for finetune '''
netparams_fixed = caffe.proto.caffe_pb2.NetParameter()
netparams_fixed.layer.add()
netparams_fixed.layer[-1].name = 'data'
netparams_fixed.layer[-1].type = 'Input'
netparams_fixed.layer[-1].top.append(last_conv)
netparams_fixed.layer[-1].input_param.shape.add()
netparams_fixed.layer[-1].input_param.shape[0].dim.extend((32,) + train_feats.shape[1:])

netparams_fixed.layer.add()
netparams_fixed.layer[-1].name = 'label'
netparams_fixed.layer[-1].type = 'Input'
netparams_fixed.layer[-1].top.append('label')
netparams_fixed.layer[-1].input_param.shape.add()
netparams_fixed.layer[-1].input_param.shape[0].dim.extend((32,))
# Add all layers after fc8
flag = False
for l in netparams.layer:
    if 'fc8' in l.name:
        l.param[0].lr_mult = 10
        l.param[0].decay_mult = 1
        l.param[1].lr_mult = 20
        l.param[1].decay_mult = 0
        l.inner_product_param.weight_filler.std = 0.001
        l.inner_product_param.bias_filler.value = 0
    if flag:
        netparams_fixed.layer.add()
        netparams_fixed.layer[-1].MergeFrom(l)
    flag = flag or l.name == chop_off_layer


# In[42]:
iter_per_epoch = int(num_images[0]/32)
# Solver
solverfile = 'ft_fixed.solver'
params = caffe.proto.caffe_pb2.SolverParameter()
params.net = 'ft_fixed.prototxt'
params.test_initialization = False
params.base_lr = 0.1
params.display = 100
params.max_iter = 240 * iter_per_epoch
params.lr_policy = "step"
params.gamma = 0.25
params.stepsize = 20*iter_per_epoch
params.momentum = 0.9
params.weight_decay = 0.000005
params.snapshot = 10000000
params.snapshot_prefix = "ft_fixed"
params.iter_size = 1
assert params.iter_size > 0
open(solverfile,'w').write(google.protobuf.text_format.MessageToString(params))
open(params.net,'w').write(google.protobuf.text_format.MessageToString(netparams_fixed))

''' Finetuning module '''
solver = caffe.get_solver('ft_fixed.solver')
solver.net.copy_from('../config_cbp_res/model_init')
for it in range(params.max_iter):
    train_ids = random.sample(range(train_feats.shape[0]),32)
    solver.net.blobs[last_conv].data[...] = train_feats[train_ids,...]
    solver.net.blobs['label'].data[...] = train_labels[train_ids]
    solver.step(1)

solver.net.save('model_init')
del solver

solver = caffe.get_solver('adapooling.solver')
solver.net.copy_from(init_weights)
solver.net.copy_from('model_init')
solver.net.save('model_init')
print('Over!')
