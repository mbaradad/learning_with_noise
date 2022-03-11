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

import resource

# to avoid ancdata error for too many open files, same as ulimit in console
# maybe not necessary, but doesn't hurt
resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))

import sys
import os
sys.path.append('.')

import vtab_eval.task_adaptation.loop as loop

def setup_environment_gpus(gpu, memory_limit=None):
    sys.path.append('.')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # export LD_LIBRARY_PATH=/data/vision/torralba/movies_sfm/home/Downloads/cuda/lib64
    # tensorflow_datasets has to

    # Just use whatever works
    CUDA = "/usr/local/cuda-10.0/lib64"
    CUDA_10_0 = "/usr/local/cuda-10.0/lib64"
    CUDA_10_1 = "/usr/local/cuda-10.1/lib64"
    CUDA_10_2 = "/usr/local/cuda-10.2/lib64"
    CUDA_11 = "/usr/local/cuda-11/lib64"

    CUDNN = "/data/vision/torralba/movies_sfm/home/programs/cuda/lib64"
    if not CUDNN in os.environ["LD_LIBRARY_PATH"]:
        raise Exception("LD_LIBRARY_PATH is not properly set!")

    import tensorflow as tf

    gpu_available = tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    )
    assert gpu_available, "GPU not available! Check missing libraries! Probably you need to setup \n" \
                          "LD_LIBRARY_PATH correctly (e.g. LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64)\n" \
                          "current is {}".format(os.environ["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in os.environ.keys() else 'ot defined!')

from absl import app
from absl import flags

FLAGS = flags.FLAGS


flags.DEFINE_bool("run_adaptation", True, "Run adaptation.")
flags.DEFINE_bool("run_evaluation", True, "Run evaluation.")

flags.DEFINE_string("tpu_name", None,
                    "Name of the TPU master. If None, defaults to using GPUs"
                    "and if GPUs are not present uses CPU.")

flags.DEFINE_string("hub_module", None, "Hub module to evaluate.")
flags.DEFINE_string("hub_module_signature", None,
                    "Name of the hub module signature.")
flags.DEFINE_string("finetune_layer", None, "Layer name for fine tunning.")

flags.DEFINE_string("work_dir", None, "Working directory for storing"
                                      "checkpoints, summaries, etc.")

flags.DEFINE_string("data_dir", "/data/vision/torralba/movies_sfm/home/tensorflow_datasets", "A directory to download and store data.")

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
flags.DEFINE_integer("batch_size", None, "Batch size for training.")
flags.DEFINE_integer("batch_size_eval", None,
                     "Batch size for evaluation: for the precise result should "
                     "be a multiplier of the total size of the evaluation"
                     "split, otherwise the reaminder is dropped.")

flags.DEFINE_list("input_range", "0.0,1.0",
                  "Two comma-separated float values that represent "
                  "min and max value of the input range.")

flags.DEFINE_float("initial_learning_rate", None, "Initial learning rate.")
flags.DEFINE_float("momentum", 0.9, "SGD momentum.")
flags.DEFINE_float("lr_decay_factor", 0.1, "Learning rate decay factor.")
flags.DEFINE_string("decay_steps", None, "Comma-separated list of steps at "
                    "which learning rate decay is performed.")
flags.DEFINE_integer("max_steps", None, "Total number of SGD updates.")
flags.DEFINE_integer("warmup_steps", 0,
                     "Number of step for warming up the leanring rate. It is"
                     "warmed up linearly: from 0 to the initial value.")

flags.DEFINE_bool("train_linear_only", None, "Weather to train only the linear layer or everything.")
flags.DEFINE_string("gpu", None, "GPU to use.")
flags.DEFINE_bool("debug", False, "GPU to use.")

flags.DEFINE_integer("save_checkpoint_steps", 500,
                     "Number of steps between consecutive checkpoints.")

flags.mark_flag_as_required("hub_module")
flags.mark_flag_as_required("hub_module_signature")
flags.mark_flag_as_required("finetune_layer")
flags.mark_flag_as_required("work_dir")
flags.mark_flag_as_required("dataset")
flags.mark_flag_as_required("batch_size")
flags.mark_flag_as_required("batch_size_eval")
flags.mark_flag_as_required("initial_learning_rate")
flags.mark_flag_as_required("decay_steps")
flags.mark_flag_as_required("max_steps")
flags.mark_flag_as_required("train_linear_only")
flags.mark_flag_as_required("gpu")


def get_data_params_from_flags(mode):
  if FLAGS.train_examples == -1:
      FLAGS.train_examples = None
  return {
      "dataset": "data." + FLAGS.dataset,
      "dataset_train_split_name": FLAGS.dataset_train_split_name,
      "dataset_eval_split_name": FLAGS.dataset_eval_split_name,
      "shuffle_buffer_size": FLAGS.shuffle_buffer_size,
      "prefetch": FLAGS.prefetch,
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
  del argv

  setup_environment_gpus(FLAGS.gpu)

  if FLAGS.run_adaptation:
    loop.run_training_loop(
        hub_module=FLAGS.hub_module,
        hub_module_signature=FLAGS.hub_module_signature,
        work_dir=FLAGS.work_dir,
        tpu_name=FLAGS.tpu_name,
        save_checkpoints_steps=FLAGS.save_checkpoint_steps,
        optimization_params=get_optimization_params_from_flags(),
        data_params=get_data_params_from_flags("adaptation"))
  if FLAGS.run_evaluation:
    loop.run_evaluation_loop(
        hub_module=FLAGS.hub_module,
        hub_module_signature=FLAGS.hub_module_signature,
        work_dir=FLAGS.work_dir,
        tpu_name=FLAGS.tpu_name,
        save_checkpoints_steps=FLAGS.save_checkpoint_steps,
        optimization_params=get_optimization_params_from_flags(),
        data_params=get_data_params_from_flags("evaluation"))


if __name__ == "__main__":
  app.run(main)