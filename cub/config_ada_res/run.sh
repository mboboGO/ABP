#!/bin/bash

source /ghome/minsb/.py2.7
cd /ghome/minsb/adapooling_cub/config_ada_res
python prepare_adapooling.py \
    --init_weights ../pretrained_models/resnet50/resnet50_iter_320000.caffemodel \
    --gpu_id 0 \
    --save_path /gdata/minsb/CUB_200_2011/models/ada_res \
    --num_classes 201 \
    --image_root /gdata/minsb/CUB_200_2011/CUB_200_2011/CUB_200_2011/images/ \
    --chop_off_layer last_relu \
    --train_batch_size 6 \
    --architecture resnet50 \
    ../data/train_imagelist.txt \
    ../data/val_imagelist.txt
