#!/bin/bash

source /ghome/minsb/.py2.7
cd /ghome/minsb/adapooling_car/config_ada_res
python prepare_adapooling.py \
    --init_weights ../pretrained_models/resnet50/resnet50_iter_320000.caffemodel \
    --gpu_id 0 \
    --save_path /gdata/minsb/car_196/models/ada_res \
    --num_classes 196 \
    --image_root /gdata/minsb/car_196/ \
    --chop_off_layer last_relu \
    --train_batch_size 6 \
    --architecture resnet50 \
    ../data/train.txt \
    ../data/test.txt
