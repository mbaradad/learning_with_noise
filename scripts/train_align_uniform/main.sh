#!/bin/bash
mkdir -p encoders
mkdir -p encoders/small_scale

GPU_0=$1
GPU_1=$2

re='^[0-9]+$'
if ! [[ $GPU_0 =~ $re ]] ; then
   echo "error: No GPU provided as argument" >&2; exit 1
fi
if ! [[ $GPU_1 =~ $re ]] ; then
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

for DATASET in ${datasets[@]}
do

    if [ ! -f encoders/small_scale/$DATASET/encoder.pth ]; then
        echo "Training encoder for dataset $DATASET"
        python align_uniform/main.py --imagefolder data/small_scale/$DATASET --result encoders/small_scale/$DATASET --gpus $GPU_0 $GPU_1
    else
        echo "Encoder already found for $DATASET, main will not run again!"
    fi
done
