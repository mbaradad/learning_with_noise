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

datasets=('sun397')

GPU=3
HUB_MODEL=https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3
HUB_MODEL_SIGNATURE=image_feature_vector
WORK_DIR=vtab_eval/results/test_default_model
FINETUNE_LAYER=resnet_v2_50/global_pool
TRAIN_LINEAR_ONLY=False

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
