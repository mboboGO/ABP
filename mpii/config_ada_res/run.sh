#!/bin/bash

source /ghome/minsb/.py2.7
cd /ghome/minsb/adapooling_mpii/config_ada_res
python prepare_adapooling.py \
    --init_weights ../pretrained_models/resnet50/resnet50_iter_320000.caffemodel \
    --gpu_id 0 \
    --num_classes 393 \
    --save_path /gdata/minsb/mpii/models/ada_res \
    --image_root /gdata/minsb/mpii/images/ \
    --chop_off_layer last_relu \
    --train_batch_size 6 \
    --architecture resnet50 \
    ../data/trainval_train.txt \
    ../data/trainval_val.txt
