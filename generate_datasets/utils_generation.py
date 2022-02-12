import numpy as np
import random
import cv2
from tqdm import tqdm
import shutil
import argparse

import os
import time, datetime

from multiprocessing import Queue, Process

def cv2_imwrite(im, file, normalize=False, jpg_quality=None):
  if len(im.shape) == 3 and im.shape[0] == 3 or im.shape[0] == 4:
    im = im.transpose(1, 2, 0)
  if normalize:
    im = (im - im.min())/(im.max() - im.min())
    im = np.array(255.0*im, dtype='uint8')
  if jpg_quality is None:
    # The default jpg quality seems to be 95
    if im.shape[-1] == 3:
      cv2.imwrite(file, im[:,:,::-1])
    else:
      raise Exception('Alpha not working correctly')
      im_reversed = np.concatenate((im[:,:,3:0:-1], im[:,:,-2:-1]), axis=2)
      cv2.imwrite(file, im_reversed)
  else:
    cv2.imwrite(file, im[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])

def cv2_resize(image, target_height_width, interpolation=cv2.INTER_NEAREST):
  if len(image.shape) == 2:
    return cv2.resize(image, target_height_width[::-1], interpolation=interpolation)
  else:
    return cv2.resize(image.transpose((1, 2, 0)), target_height_width[::-1], interpolation=interpolation).transpose((2, 0, 1))

def cv2_imread(file, return_BGR=False):
  im = cv2.imread(file)
  if im is None:
    raise Exception('Image {} could not be read!'.format(file))
  im = im.transpose(2,0,1)
  if return_BGR:
    return im
  return im[::-1, :, :]

def listdir(folder, prepend_folder=False, extension=None, type=None):
  assert type in [None, 'file', 'folder'], "Type must be None, 'file' or 'folder'"
  files = [k for k in os.listdir(folder) if (True if extension is None else k.endswith(extension))]
  if type == 'folder':
    files = [k for k in files if os.path.isdir(folder + '/' + k)]
  elif type == 'file':
    files = [k for k in files if not os.path.isdir(folder + '/' + k)]
  if prepend_folder:
    files = [folder + '/' + f for f in files]
  return files

def str2bool(v):
  assert type(v) is str
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean (yes, true, t, y or 1, lower or upper case) string expected.')

def select_gpus(gpus_arg):
  #so that default gpu is one of the selected, instead of 0
  gpus_arg = str(gpus_arg)
  if len(gpus_arg) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_arg
    gpus = list(range(len(gpus_arg.split(','))))
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    gpus = []
  print('CUDA_VISIBLE_DEVICES={}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

  flag = 0
  for i in range(len(gpus)):
    for i1 in range(len(gpus)):
      if i != i1:
        if gpus[i] == gpus[i1]:
          flag = 1
  assert not flag, "Gpus repeated: {}".format(gpus)

  return gpus


class ThreadeImageWriter():
  # plot func receives a dict and gets what it needs to plot
  def __init__(self, queue_size, use_threading=True, force_except=False):
    self.queue = Queue(queue_size)
    self.use_threading = use_threading
    self.force_except = force_except
    def plot_results_process(queue):
        # to avoid wasting time making videos
        while True:
            try:
                if queue.empty():
                    time.sleep(1)
                    if queue.full():
                        print("Writing queue is full!")
                else:
                    to_write = queue.get()
                    self.write_image(*to_write)
                    continue
            except Exception as e:
                if self.force_except:
                  raise e
                print('Plotting failed wiht exception: ')
                print(e)
    if self.use_threading:
      Process(target=plot_results_process, args=[self.queue]).start()

  def write_image(self, images, files):
      for i, image_file in enumerate(files):
          if os.path.exists(image_file):
              continue
          cv2_imwrite(images[i], image_file, normalize=True)

  def put_images_and_files(self, images, files):
    try:
      if self.use_threading:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
        self.queue.put((images, files))
      else:
        self.write_image(images, files)
    except Exception as e:
      if self.force_except:
        raise e
      print('Putting onto plot queue failed with exception:')
      print(e)

