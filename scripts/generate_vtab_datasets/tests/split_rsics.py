from my_python_utils.common_utils import *

import os

rsics_manual_path = '/data/vision/torralba/movies_sfm/home/tensorflow_datasets/downloads/manual/NWPU-RESISC45'
for full_filename in tqdm(sorted(listdir(rsics_manual_path, prepend_folder=True, type='file', extension='.jpg'))):
    filename = os.path.basename(full_filename)
    photo_name = filename.split('_')[-1]
    object_type = '_'.join(filename.split('_')[:-1])
    folder = rsics_manual_path + '/' + object_type
    os.makedirs(folder, exist_ok=True)
    shutil.move(full_filename, folder + '/' + photo_name)