#!/bin/bash
#SBATCH -J train_imagenet_stylegan2_4GPU
#SBATCH -o "$HOME/slurm_logs/train_imagenet_stylegan2_4GPU_%j"
#SBATCH --mail-user=mbaradad@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:volta:1
#SBATCH -N 1
#SBATCH -n 40

echo " Running on single nodes with 4 GPU"
echo ""
echo " Run started at:- "
date

STYLEGAN_PATH=/home/gridsan/mbaradad/projects/no_training_cnns/contrastive_image_models/image_generation/stylegan/stylegan2-ada-pytorch
cd $STYLEGAN_PATH

DATA_DIR=$IMAGENET_PATH/train

OUT_DIR=imagenet_dir
FOUND=false
if [ -d $OUT_DIR ]; then
    CHECKPOINTS=($(ls --color=none -d $OUT_DIR/*/*.pkl | sort))
    if [ ${#CHECKPOINTS[@]} -gt 0 ]; then

        echo $CHECKPOINTS

        echo "Number of checkpoints: ${#CHECKPOINTS[@]}"
        LATEST_CHECKPOINT=${CHECKPOINTS[${#CHECKPOINTS[@]}-1]}

        echo "Latest checkpoint: $LATEST_CHECKPOINT"

        LATEST_CHECKPOINT_KIMG=1
        FOUND=true

        arrIN=(${LATEST_CHECKPOINT//-/ })
        RESUME_KIMG_TMP=${arrIN[${#arrIN[@]}-1]}

        arrIN=(${RESUME_KIMG_TMP//./ })
        RESUME_KIMG=${arrIN[0]}

    fi
fi

if [ "$FOUND" = true ] ; then
    echo "Previous checkpoint found! Will restart training!"
    echo "Resuming from: $LATEST_CHECKPOINT"
    sleep 1s
    python train.py --outdir=$OUT_DIR --data=$DATA_DIR --gpus=4 --resume=$LATEST_CHECKPOINT --resume-kimg=$RESUME_KIMG

else
    echo "Previous checkpoint not found!"
    python train.py --outdir=$OUT_DIR --data=$DATA_DIR --gpus=4
fi


echo "Run completed at:- "
date

# to excecute, run:
# sbatch stylegan_imagenet.slurm