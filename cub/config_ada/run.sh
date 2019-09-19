#!/bin/bash

source /ghome/minsb/.py2.7
cd /ghome/minsb/adapooling_cub/config_ada
python prepare_adapooling.py \
    --init_weights ../pretrained_models/vgg16/vgg16_imagenet.caffemodel \
    --gpu_id 0 \
    --num_classes 201 \
    --image_root /gdata/minsb/CUB_200_2011/CUB_200_2011/CUB_200_2011/images/ \
    --save_path /gdata/minsb/CUB_200_2011/models/ada \
    --chop_off_layer relu5_3 \
    --train_batch_size 8 \
    --architecture vgg16 \
    ../data/train_imagelist.txt \
    ../data/val_imagelist.txt
