#!/bin/bash

# imagenet_100 path ours:
# /data/vision/torralba/datasets/imagenet100

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
for MODEL in "${models[@]}"; do
    ACCURACY=$( tail -n 1 lincls_imagenet/small_scale/$DATASET/log_eval.txt )
    echo "$DATASET: $ACCURACY"
done