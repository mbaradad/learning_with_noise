#MODEL=$1
#TRAIN_LINEAR_ONLY=$2
#FULL=$3
#GPU=$4

./vtab_eval/experiments/test_model.sh moco_v2_pretrained_200 True False 3

#./vtab_eval/experiments/test_model.sh moco_v2_pretrained_200 True True 3
#./vtab_eval/experiments/test_model.sh moco_v2_pretrained_200 False False 3
#./vtab_eval/experiments/test_model.sh moco_v2_pretrained_200 False True 3