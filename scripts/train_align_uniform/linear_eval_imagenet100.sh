#!/bin/bash

# imagenet_100 path ours:
# /data/vision/torralba/datasets/imagenet100

mkdir -p encoders
mkdir -p encoders/small_scale

IMAGENET_PATH=$1
GPU_0=$2

re='^[0-9]+$'
if ! [[ $GPU_0 =~ $re ]] ; then
   echo "error: No GPU provided as argument" >&2; exit 1
fi

datasets=('dead_leaves-squares' \
          'dead_leaves-oriented' \
          'dead_leaves-textures' \
          'dead_leaves-mixed' \

          'stat-spectrum' \
          'stat-wmm' \
          'stat-spectrum_color' \
          'stat-spectrum_color_wmm' \

          'stylegan-random' \
          'stylegan-highfreq' \
          'stylegan-sparse' \
          'stylegan-oriented' \

          'feature_vis-random' \
          'feature_vis-dead_leaves' \
           )

# Imagenet100
for DATASET in "${datasets[@]}"; do
    if [ ! -f lincls_imagenet/small_scale/$DATASET/val_acc.txt ]; then
        echo "Running Imagenet100 for $DATASET"
        CUDA_VISIBLE_DEVICES=$GPU_0 python align_uniform/linear_eval_imagenet100.py -d $DATASET --imagenet100_path $IMAGENET_PATH
    else
        echo "Imagenet100 linear evaluation already computed for $DATASET, main will not run again!"
    fi
done