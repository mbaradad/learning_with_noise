import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from my_python_utils.common_utils import *

dumped_datasets_path='/data/vision/torralba/scratch/mbaradad/vtab_datasets'
datasets = listdir(dumped_datasets_path, prepend_folder=True)

completed_datasets = listdir(dumped_datasets_path + '/completed_datasets_list', prepend_folder=False)
completed_datasets.sort()

for dataset in completed_datasets:
  print("Testing number of files for dataset: " + dataset)
  for split in ['train', 'val']:
    dataset_split_path = '{}/{}/{}'.format(dumped_datasets_path, dataset, split)
    files = find_all_files_recursively(dataset_split_path)
    print("{}: {}".format(split.capitalize(), len(files)))