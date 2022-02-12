#!/usr/bin/env bash
EXP_NAME=$1
IMAGENET_FOLDER=$2

RESULTS_FOLDER=encoders/large_scale/$EXP_NAME

echo "Computing main moco!"
python moco/main_moco.py --batch-size 256 --mlp --moco-t 0.2 --aug-plus --cos --multiprocessing-distributed --world-size 1 --rank 0 --dist-url tcp://localhost:10043 \
--epochs 200 --restart-latest --result-folder $RESULTS_FOLDER \
--dataset_type imagefolder --log-wandb True --workers 65 \
data/large_scale/$EXP_NAME

echo "Computing linear eval!"
python moco/main_lincls.py -a resnet50 --batch-size 256 \
  --dist-url 'tcp://localhost:10043' --multiprocessing-distributed --world-size 1 --rank 0 --lr 30.0 \
  --pretrained $RESULTS_FOLDER/checkpoint_0199.pth.tar \
  --restart-latest \
  --result-folder $RESULTS_FOLDER/main_lincls/epoch_199 \
  $IMAGENET_FOLDER