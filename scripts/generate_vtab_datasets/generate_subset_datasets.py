import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from my_python_utils.common_utils import *
from p_tqdm import p_map

TOTAL_SAMPLES = 10000

vtab_dumped_path = '/data/vision/torralba/movies_sfm/home/no_training_cnn/contrastive_image_models/vtab_datasets'
vtab_subset_path = '/data/vision/torralba/movies_sfm/home/no_training_cnn/contrastive_image_models/vtab_datasets_{}_subset'.format(TOTAL_SAMPLES)

completed_datasets = vtab_dumped_path + '/completed_datasets_list'
datasets = listdir(completed_datasets)

def generate_one(split_dataset, check_all_generated=False, sample_at_random=True):
  random.seed(1337)
  np.random.seed(1337)
  split, dataset = split_dataset
  generated_split_path = '{}/{}/{}'.format(vtab_subset_path, dataset, split)
  completed_folder = '{}/completed_datasets'.format(vtab_subset_path)
  os.makedirs(completed_folder, exist_ok=True)

  original_split_path = '{}/{}/{}'.format(vtab_dumped_path, dataset, split)
  classes_paths = listdir(original_split_path, prepend_folder=True)
  classes_paths.sort()

  completed_dataset_file = '{}/{}'.format(completed_folder, dataset + '_' + split)
  if os.path.exists(generated_split_path):
    if os.path.exists(completed_dataset_file):
      return
    subset_images = find_all_files_recursively(generated_split_path, extension='.jpg')
    if len(subset_images) == TOTAL_SAMPLES:
      touch(completed_dataset_file)
      print("Datset {} already generated!".format(generated_split_path))
      return
    elif len(subset_images) > TOTAL_SAMPLES:
      print("More samples than desired: {} vs {}".format(len(subset_images), TOTAL_SAMPLES))
      print("For dataset: " + generated_split_path)
      exit(0)
    elif check_all_generated:
      original_dataset_samples = find_all_files_recursively(original_split_path, extension='.jpg')
      if len(original_dataset_samples) == len(subset_images):
        touch(completed_dataset_file)
      else:
        print("Less samples than expected and check_all_generated is True {} vs {}! Regenerate the dataset or check the generation code is ok!".format(len(subset_images), TOTAL_SAMPLES))
        print("For dataset with {} samples: ".format() + generated_split_path)
        exit(0)
  elif os.path.exists(completed_dataset_file):
    delete_file(completed_dataset_file)

  if sample_at_random:
    # sample at random so that the weights per class are the same
    all_imgs = find_all_files_recursively(original_split_path, extension='.jpg', prepend_folder=True)
    if len(all_imgs) > TOTAL_SAMPLES:
      imgs = random.sample(all_imgs, TOTAL_SAMPLES)
    else:
      imgs = all_imgs
    for img in tqdm(imgs):
      current_class = img.split('/')[-2]
      class_output_path = '{}/{}'.format(generated_split_path, current_class)
      os.makedirs(class_output_path, exist_ok=True)
      dst_path = '{}/{}'.format(class_output_path, img.split('/')[-1])
      if not os.path.exists(dst_path):
        shutil.copy(img, dst_path)
  else:
    # sample the same number per class
    n_classes = len(classes_paths)

    ceil = int(np.ceil(TOTAL_SAMPLES / n_classes))
    floor = int(np.floor(TOTAL_SAMPLES / n_classes))
    n_images_per_class = [ceil for _ in range(TOTAL_SAMPLES % n_classes)] + [floor for _ in range(n_classes - TOTAL_SAMPLES % n_classes)]

    assert len(n_images_per_class) == n_classes, "Different number of n_images_per_class {} and n_classes {}.".format(n_images_per_class, n_classes)
    assert all([k > 0 for k in n_images_per_class]), "Some classes have 0 images! {}".format(n_images_per_class)
    assert sum(n_images_per_class) == TOTAL_SAMPLES, "Not the desired number of n_images_per_class, check the code!"

    n_images_and_class_path = list(zip(n_images_per_class, classes_paths))
    for current_n_images, current_class_path in n_images_and_class_path:
      class_output_path = '{}/{}'.format(generated_split_path, current_class_path.split('/')[-1])
      os.makedirs(class_output_path, exist_ok=True)
      all_imgs = listdir(current_class_path, prepend_folder=True)
      if len(all_imgs) > current_n_images:
        imgs = random.sample(all_imgs, current_n_images)
      else:
        imgs = all_imgs
      for img in tqdm(imgs):
        dst_path = '{}/{}'.format(class_output_path, img.split('/')[-1])
        if not os.path.exists(dst_path):
          shutil.copy(img, dst_path)

def check_one(split_dataset):
  split, dataset = split_dataset

  output_dir = '{}/{}/{}'.format(vtab_subset_path, dataset, split)
  all_files = find_all_files_recursively(output_dir)
  assert len(all_files) == TOTAL_SAMPLES, "Found {} files but expected {}".format(len(all_files), TOTAL_SAMPLES)


to_generate = []
for dataset in datasets:
  for split in ['train', 'val']:
    to_generate.append((split, dataset))

parallel = True
just_check = True

process_in_parallel_or_not(lambda x: generate_one(x, check_all_generated=just_check), to_generate, parallel)