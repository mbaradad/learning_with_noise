#MODEL=$1
#TRAIN_LINEAR_ONLY=$2
#FULL=$3
#GPU=$4

./vtab_eval/experiments/test_model.sh resnet50_random True False 1

#./vtab_eval/experiments/test_model.sh resnet50_random True True 1
#./vtab_eval/experiments/test_model.sh resnet50_random False False 1
#./vtab_eval/experiments/test_model.sh resnet50_random False True 1