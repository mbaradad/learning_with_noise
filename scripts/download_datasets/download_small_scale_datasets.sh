#!/bin/bash
mkdir -p data
mkdir -p data/small_scale

datasets=('dead_leaves-squares' \
          'dead_leaves-oriented' \
          'dead_leaves-mixed' \
          'dead_leaves-textures' \

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
    echo "Downloading $DATASET"
    wget -O data/small_scale/$DATASET.zip http://data.csail.mit.edu/noiselearning/zipped_data/small_scale/$DATASET.zip
    yes | unzip data/small_scale/$DATASET.zip -d data/small_scale/$DATASET
    rm data/small_scale/$DATASET.zip
done
