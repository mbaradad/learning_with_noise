#MODEL=$1
#TRAIN_LINEAR_ONLY=$2
#FULL=$3
#GPU=$4

./vtab_eval/experiments/test_model.sh moco_v2_imagenet_pytorch True False 0
./vtab_eval/experiments/test_model.sh moco_v2_my_dead_leaves_mixed_large_res_256_RGB True False 0
./vtab_eval/experiments/test_model.sh resnet50_random True False 0