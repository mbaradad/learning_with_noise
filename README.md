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
To download all the datasets, first run:
```
./scripts/download_datasets/download_small_scale_datasets.sh
```

Then you can launch the contrastive training for all the small scale experiments with:
```
./scripts/train_align_uniform/main.sh <GPU_ID_0> <GPU_ID_1>
```
If you just want to test the linear evaluation of the models (or do something else with them), you can directly download our pretrained encoders with:
```
./scripts/download_pretrained_models/download_all_alexnet_encoders.sh
```

Finally, you can evaluate the linear performance with imagenet100 as:

```
./scripts/train_align_uniform/linear_eval.sh <path-to-imagenet100> <GPU_ID>
```
Where <path-to-imagenet100> is the path to the imagenet100 dataset dir, which should contain two dirs (train and val) each with the train and val samples respectively for the 100 imagenet100 classes. 
If you have imagenet1k, you can generate imagenet100 using the following command (which will create simlyncs to your imagenet1k dir):

```
./scripts/datasets/generate_imagenet100.sh <path-to-imagenet1k> <path-to-imagenet100>
```


# Large scale experiments
Datasets and encoders will be be released soon!
<!-- 
To reproduce the large scale experiments, first download all datasets with:
```
./scripts/download_datasets/download_large_scale_datasets.sh
```
-->

# Data generation
Scripts to generate the datasets will be released soon!
<!--
To replicate the data generation processes, see the dataset_generation/README.md
-->
  
# Citation
```
@inproceedings{baradad2021learning,
  title={Learning to See by Looking at Noise},
  author={Manel Baradad and Jonas Wulff and Tongzhou Wang and Phillip Isola and Antonio Torralba},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
