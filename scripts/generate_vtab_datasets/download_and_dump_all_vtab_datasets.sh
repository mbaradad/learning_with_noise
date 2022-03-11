#!/bin/bash
datasets=('caltech101' \
          'cifar(num_classes=100)' \
          'clevr(task="closest_object_distance")' \
          'clevr(task="count_all")' \
          'diabetic_retinopathy(config="btgraham-300")'\
          'dmlab' \
          'dsprites(predicted_attribute="label_x_position",num_classes=16)' \
          'dsprites(predicted_attribute="label_orientation",num_classes=16)' \
          'dtd' \
          'eurosat' \
          'kitti(task="closest_vehicle_distance")' \
          'oxford_iiit_pet' \
          'oxford_flowers102' \
          'patch_camelyon' \
          'resisc45' \
          'smallnorb(predicted_attribute="label_azimuth")' \
          'smallnorb(predicted_attribute="label_elevation")' \
          'sun397_ours'\
          'svhn'
          )


DATA_DIR=/tmp
DUMP_PATH=../../vtab_datasets

for DATASET in "${datasets[@]}"
do
    echo "Processing dataset $DATASET"
    python dump_datasets.py --dataset \"$DATASET\" --data_dir $DATA_DIR, --dump_datasets_path $DUMP_PATH
    # process in parallel
    # python dump_datasets.py --dataset \"$DATASET\" &
done
wait

