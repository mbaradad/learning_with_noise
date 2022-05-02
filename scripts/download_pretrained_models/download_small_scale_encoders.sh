#!/bin/bash
mkdir -p encoders
mkdir -p encoders/small_scale

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

for DATASET in ${datasets[@]}
do
    echo "Downloading pretrained model for dataset $DATASET"
    mkdir -p encoders/small_scale/$DATASET
    wget -O encoders/small_scale/$DATASET/encoder.pth http://data.csail.mit.edu/noiselearning/encoders/small_scale/$DATASET/encoder.pth

done
