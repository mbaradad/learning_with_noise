#!/bin/bash
# coding=utf-8
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script fine tunes a given Hub model on all tasks from the VTAB benchmark.
# It expects two parameters:
# 1. Hub model path or link.
# 2. Work directory path.

datasets=('caltech101' \
          'cifar(num_classes=100)' \
          'dtd' \
          'oxford_flowers102' \
          'oxford_iiit_pet' \
          'patch_camelyon' \
          'svhn' \
          'resisc45' \
          'eurosat' \
          'dmlab' \
          'kitti(task="closest_vehicle_distance")' \
          'smallnorb(predicted_attribute="label_azimuth")' \
          'smallnorb(predicted_attribute="label_elevation")' \
          'dsprites(predicted_attribute="label_x_position",num_classes=16)' \
          'dsprites(predicted_attribute="label_orientation",num_classes=16)' \
          'clevr(task="closest_object_distance")' \
          'clevr(task="count_all")' \
          'diabetic_retinopathy(config="btgraham-300")')

# to add when too many files open error is solved, removed for now to make training faster
# 'sun397' \

TRAIN_LINEAR_ONLY=False
FULL=False
GPU=$1

HUB_MODEL=https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3
HUB_MODEL_SIGNATURE=image_feature_vector
WORK_DIR=vtab_eval/results/resnet_v2_tfhub
FINETUNE_LAYER=resnet_v2_50/global_pool

if [ $TRAIN_LINEAR_ONLY == "True" ]; then
    echo "Train linear only!"
    WORK_DIR="${WORK_DIR}_linear_only"
else
    echo "Finetuning everything!"
fi

if [ $FULL == "True" ]; then
    echo "Train all samples!"
    WORK_DIR="${WORK_DIR}_all_samples"
    TRAIN_SAMPLES=-1
else
    echo "Train with 1K samples"
    WORK_DIR="${WORK_DIR}_1k_samples"
    TRAIN_SAMPLES=1000
fi

mkdir -p $WORK_DIR

LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH

adapt_and_eval(){
    gpu=$1
    dataset=$2
    hub=$3
    hub_module_signature=$4
    workdir=$5
    finetune_layer=$6
    train_linear_only=$7
    python vtab_eval/task_adaptation/adapt_and_eval.py \
      --hub_module $hub \
      --hub_module_signature $hub_module_signature \
      --finetune_layer $finetune_layer \
      --train_linear_only $train_linear_only \
      --work_dir $workdir/$dataset \
      --dataset $dataset \
      --train_examples 1000 \
      --batch_size 64 \
      --batch_size_eval 10 \
      --initial_learning_rate 0.01 \
      --decay_steps 300,600,900 \
      --input_range 0.0,1.0 \
      --max_steps 1000 \
      --warmup_steps 80 \
      --save_checkpoint_steps 1000 \
      --gpu $gpu
}

for dataset in "${datasets[@]}"
do
    adapt_and_eval $GPU $dataset $HUB_MODEL $HUB_MODEL_SIGNATURE $WORK_DIR $FINETUNE_LAYER $TRAIN_LINEAR_ONLY
done
