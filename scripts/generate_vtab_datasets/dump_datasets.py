#!/usr/bin/python
# coding=utf-8
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A script for running hub-module adaptation and evaluaton."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append('..')
sys.path.append('../..')
sys.path.append(os.path.dirname(os.path.realpath(sys.argv[0])))

import resource

# to avoid ancdata error for too many open files, same as ulimit in console
# maybe not necessary, but doesn't hurt
resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))

from my_python_utils.common_utils import *

import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from p_tqdm import p_map

import vtab_eval.task_adaptation.loop as loop
from vtab_eval.task_adaptation.data_loader import build_data_pipeline


from absl import app
from absl import flags

from collections import defaultdict

FLAGS = flags.FLAGS

# to run just use the default evaluation comand with an arbitrary hub_model with each of the datasets
flags.DEFINE_bool("run_adaptation", True, "Run adaptation.")
flags.DEFINE_bool("run_evaluation", True, "Run evaluation.")

flags.DEFINE_string("tpu_name", None,
                    "Name of the TPU master. If None, defaults to using GPUs"
                    "and if GPUs are not present uses CPU.")

flags.DEFINE_string("data_dir", "/data/vision/torralba/movies_sfm/home/scratch/mbaradad/tensorflow_datasets", "A directory to download and store data.")
flags.DEFINE_string("dump_datasets_path", "/data/vision/torralba/scratch/mbaradad/vtab_datasets", "A directory where the datasets will be dumped.")


flags.DEFINE_string("dataset", None, "Dataset name.")
flags.DEFINE_enum("dataset_train_split_name", "trainval",
                  ["train", "val", "trainval", "test"],
                  "Dataset train split name.")
flags.DEFINE_enum("dataset_eval_split_name", "test",
                  ["train", "val", "trainval", "test"],
                  "Dataset evaluation split name.")
flags.DEFINE_integer("shuffle_buffer_size", 10000,
                     "A size of the shuffle buffer.")
flags.DEFINE_integer("prefetch", 1000,
                     "How many batches to prefetch in the input pipeline.")
flags.DEFINE_integer("train_examples", None,
                     "How many training examples to use. Defaults to all.")
flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
flags.DEFINE_integer("batch_size_eval", 10,
                     "Batch size for evaluation: for the precise result should "
                     "be a multiplier of the total size of the evaluation"
                     "split, otherwise the reaminder is dropped.")

flags.DEFINE_list("input_range", "0.0,1.0",
                  "Two comma-separated float values that represent "
                  "min and max value of the input range.")

flags.DEFINE_float("initial_learning_rate", 1e-2, "Initial learning rate.")
flags.DEFINE_float("momentum", 0.9, "SGD momentum.")
flags.DEFINE_float("lr_decay_factor", 0.1, "Learning rate decay factor.")
flags.DEFINE_string("decay_steps", "300,600,900", "Comma-separated list of steps at "
                    "which learning rate decay is performed.")
flags.DEFINE_integer("max_steps", 10, "Total number of SGD updates.")
flags.DEFINE_integer("warmup_steps", 0,
                     "Number of step for warming up the leanring rate. It is"
                     "warmed up linearly: from 0 to the initial value.")

flags.DEFINE_bool("debug", False, "GPU to use.")

flags.DEFINE_integer("save_checkpoint_steps", 500,
                     "Number of steps between consecutive checkpoints.")

flags.mark_flag_as_required("dataset")
#flags.mark_flag_as_required("batch_size")
#flags.mark_flag_as_required("batch_size_eval")
#flags.mark_flag_as_required("initial_learning_rate")
#flags.mark_flag_as_required("decay_steps")
#flags.mark_flag_as_required("max_steps")


def get_data_params_from_flags(mode):
  # override so that everything is dumped
  FLAGS.train_examples = None
  return {
      "dataset": "data." + FLAGS.dataset,
      "dataset_train_split_name": FLAGS.dataset_train_split_name,
      "dataset_eval_split_name": FLAGS.dataset_eval_split_name,
      "shuffle_buffer_size": FLAGS.shuffle_buffer_size,
      "prefetch": 30,
      "train_examples": FLAGS.train_examples,
      "batch_size": FLAGS.batch_size,
      "batch_size_eval": FLAGS.batch_size_eval,
      "data_for_eval": mode == "adaptation",
      "data_dir": FLAGS.data_dir,
      "input_range": [float(v) for v in FLAGS.input_range]
  }


def get_optimization_params_from_flags():
  return {
      "train_linear_only": FLAGS.train_linear_only,
      "finetune_layer": FLAGS.finetune_layer,
      "initial_learning_rate": FLAGS.initial_learning_rate,
      "momentum": FLAGS.momentum,
      "lr_decay_factor": FLAGS.lr_decay_factor,
      "decay_steps": [int(x) for x in FLAGS.decay_steps.split(",")],
      "max_steps": FLAGS.max_steps,
      "warmup_steps": FLAGS.warmup_steps,
      "tpu_name": FLAGS.tpu_name,
      "debug": FLAGS.debug
  }


def main(argv):
  if FLAGS.dataset[0] in ['\'', '"']:
    FLAGS.dataset = FLAGS.dataset[1:]
  if FLAGS.dataset[-1] in ['\'', '"']:
    FLAGS.dataset = FLAGS.dataset[:-1]

  dataset_name = FLAGS.dataset
  if dataset_name == 'sun397':
    raise Exception("sun397 fails on one image, which makes the full dataset generation fail. Use sun397_ours instead")

  completed_path = FLAGS.dump_datasets_path + '/completed_datasets_list/' + dataset_name
  if os.path.exists(completed_path):
    print("Dataset {} already dumped. Will exit!".format(dataset_name))
    exit(1)

  train_data_params = get_data_params_from_flags("adaptation")
  eval_data_params = get_data_params_from_flags("evaluation")

  print("Dumping dataset: {}".format(dataset_name))
  output_path = FLAGS.dump_datasets_path + '/' + dataset_name

  os.makedirs(os.path.dirname(completed_path), exist_ok=True)

  batch_size = 1
  # for eval disables shuffling
  # no_preprocess True disables preprocessing for neural net
  try:
    train_dataset, n_train_examples = build_data_pipeline(train_data_params, 'train', return_n_examples=True, for_eval=True, no_preprocess=True)
    train_dataset = train_dataset({'batch_size': batch_size})
    val_dataset, n_val_examples = build_data_pipeline(eval_data_params, 'eval', return_n_examples=True, for_eval=True, no_preprocess=True)
    val_dataset = val_dataset({'batch_size': batch_size})
  except Exception as e:
    print(e)
    print("If some error has occurred, try to delete downloaded datasets, data_dir... as this may correspond to other verison of tensorflow_dataset, "
          "and the downloaded/cached stuff is not compatible between different versions.")
    raise e

  print("Train examples: {}".format(n_train_examples))
  print("Val examples: {}".format(n_val_examples))

  train_path = output_path + '/train'
  val_path = output_path + '/val'

  os.makedirs(train_path, exist_ok=True)
  os.makedirs(val_path, exist_ok=True)

  def count_files(path):
      print("Counting files in: " + path)
      class_folders = listdir(path, prepend_folder=True)
      n_images = 0
      for class_folder in class_folders:
          n_images += len(listdir(class_folder))
      return n_images

  train_examples_computed = count_files(train_path)
  val_examples_computed = count_files(val_path)

  if train_examples_computed >= n_train_examples and val_examples_computed >= n_val_examples:
      print("Everything already computed for dataset {}!".format(dataset_name))
      touch(completed_path)
      exit(1)

  for dataset, path, dataset_examples in [(train_dataset, train_path, n_train_examples), (val_dataset, val_path, n_val_examples)]:
      written_images_per_id = defaultdict(int)
      examples_seen = 0
      for element in tqdm(dataset):
          if examples_seen >= dataset_examples:
              break
          images_to_write = []
          for k in range(batch_size):
              if examples_seen >= dataset_examples:
                  break
              img = np.array(element['image'][k]).transpose((2,0,1))
              label = np.array(element['label'])[k]
              if 'id' in element.keys():
                id = str(np.array(element['id'])[k])[2:-1]
              else:
                id = str(written_images_per_id[label]).zfill(6)
                written_images_per_id[label] += 1

              img_dir = path + '/{}'.format(str(label).zfill(4))
              os.makedirs(img_dir, exist_ok=True)

              images_to_write.append((img, img_dir + '/{}.jpg'.format(id)))

              examples_seen += 1

          def write_image(img_and_path):
              img, path = img_and_path
              if os.path.exists(path):
                return
              # normalize
              img = (img - img.min())/(img.max() - img.min())
              img = np.array(img * 255, dtype='uint8')
              cv2_imwrite(img * 255, path)

          if batch_size > 32:
            p_map(write_image, images_to_write, num_cpus=32)
          else:
            for img_to_write in images_to_write:
              write_image(img_to_write)


if __name__ == "__main__":
  app.run(main)