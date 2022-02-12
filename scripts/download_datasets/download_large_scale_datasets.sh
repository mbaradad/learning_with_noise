#!/bin/bash
mkdir -p data
mkdir -p data/large_scale

datasets=('dead_leaves-mixed' \
          'feature_vis-dead_leaves' \
          'stat-spectrum_color_wmm' \
          'stylegan-oriented' \
           )

for DATASET in ${datasets[@]}
do
    echo "Downloading $DATASET"
    wget -O data/large_scale/$DATASET.zip http://data.csail.mit.edu/noiselearning/zipped_data/large_scale/$DATASET.zip
    yes | unzip data/large_scale/$DATASET.zip -d data/large_scale/$DATASET
    rm data/large_scale/$DATASET.zip
done
