#!/bin/bash

source /ghome/minsb/.py2.7
cd /ghome/minsb/adapooling_car/config_ada
python prepare_adapooling.py \
    --init_weights ../pretrained_models/vgg16/vgg16_imagenet.caffemodel \
    --gpu_id 0 \
    --save_path /gdata/minsb/car_196/models/ada \
    --num_classes 196 \
    --image_root /gdata/minsb/car_196/ \
    --chop_off_layer relu5_3 \
    --train_batch_size 8 \
    --architecture vgg16 \
    ../data/train.txt \
    ../data/test.txt
