#!/bin/bash
mkdir -p encoders
mkdir -p encoders/small_scale

datasets=('dead_leaves-mixed' \
          'feature_vis-dead_leaves'
          'stat-spectrum_color_wmm' \
          'stylegan-oriented')

for DATASET in ${datasets[@]}
do
    echo "Downloading pretrained model for dataset $DATASET"
    mkdir -p encoders/large_scale/$DATASET
    wget -O encoders/large_scale/$DATASET/checkpoint_0199.pth.tar http://data.csail.mit.edu/noiselearning/encoders/large_scale/$DATASET/checkpoint_0199.pth.tar
done

DATASET=mixed-4
echo "Downloading pretrained model for dataset $DATASET"
mkdir -p encoders/large_scale/$DATASET
wget -O encoders/large_scale/$DATASET/checkpoint_0799.pth.tar http://data.csail.mit.edu/noiselearning/encoders/large_scale/$DATASET/checkpoint_0799.pth.tar
