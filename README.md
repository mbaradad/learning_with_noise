# Learning to See by Looking at Noise

This is the official implementation of Learning to See by Looking at Noise (NeurIPS 2021, Spotlight).

<p align="center">
  <img width="100%" src="https://mbaradad.github.io/learning_with_noise/images/teaser.jpeg">
</p>

In this work, we investigate a suite of image generation models that produce images from simple random processes. These are then used as training data for a visual representation learner with a contrastive loss. We study two types of noise processes, statistical image models and deep generative models under different random initializations. 

[[Project page](https://mbaradad.github.io/learning_with_noise/)] 
[[Paper](https://arxiv.org/pdf/2106.05963.pdf)]
[[arXiv](https://arxiv.org/abs/2106.05963)]

# Requirements
This version of code has been tested with Python 3.7.7 and pytorch 1.6.0. Other versions of pytorch are likely to work out of the box. 
The contrastive training requires two GPU's with at least 12GB of memory for the small scale experiments, while the large scale experiments require the same computation resources as the facebookresearch implementation of MoCo.


To use this repo, first clone it and correct the permissions for the scripts with:
```
git clone https://github.com/mbaradad/learning_with_noise
cd learning_with_noise
chmod 755 -R scripts
```

To install all the requirements, simply do:

```
pip intall -r requirements.txt
```

# Small scale experiments
To download all the small scale datasets, first run:
```
./scripts/download_datasets/download_small_scale_datasets.sh
```

Then you can launch the contrastive training for all the small scale experiments with:
```
./scripts/train_align_uniform/main.sh <GPU_ID_0> <GPU_ID_1>
```
If you just want to test the linear evaluation of the models (or do something else with them), you can directly download our pretrained encoders with:
```
./scripts/download_pretrained_models/download_small_scale_encoders.sh
```

Finally, you can evaluate the linear performance with imagenet100 as:

```
./scripts/train_align_uniform/linear_eval.sh <path-to-imagenet100> <GPU_ID>
```
Where <path-to-imagenet100> is the path to the imagenet100 dataset dir, which should contain two dirs (train and val) each with the train and val samples respectively for the 100 imagenet100 classes. 
If you have imagenet1k, you can generate imagenet100 using the following command (which will create simlyncs to your imagenet1k dir):

```
./scripts/generate_datasets/generate_imagenet100.sh <path-to-imagenet1k> <path-to-imagenet100>
```


# Large scale experiments
If you just want to test the linear evaluation of the models (or do something else with them), you can directly download our pretrained Resnet-50 encoders with:

```
./scripts/download_pretrained_models/download_large_scale_encoders.sh
```

To download all large scale datasets, first run:

```
./scripts/download_datasets/download_large_scale_datasets.sh
```

Then to train and evaluate MoCo v2 run, substituting `EXP_NAME` for the desired experiment name and `IMAGENET_PYTORCH` for your local copy of Imagenet-1k containing 
train/val folders:

```
EXP_NAME=stylegan-oriented
RESULTS_FOLDER=encoders/large_scale/$EXP_NAME
IMAGENET_FOLDER=your_imagenet_path

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
```

# Data generation
The scripts to generate the datasets are in `generate_datasets` folder. 
A README on how to execute them can be found in the folder.
  
# Citation
```
@InProceedings{Baradad_2018_CVPR,
author = {Baradad, Manel and Ye, Vickie and Yedidia, Adam B. and Durand, Frédo and Freeman, William T. and Wornell, Gregory W. and Torralba, Antonio},
title = {Inferring Light Fields From Shadows},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
