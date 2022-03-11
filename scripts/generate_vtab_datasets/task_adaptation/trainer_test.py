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

"""Tests for the trainer module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import task_adaptation.trainer as trainer

import tensorflow.compat.v1 as tf


class ModelTest(absltest.TestCase):

  def test_get_train_op(self):
    dummy_net = tf.Variable(0.0) + 0.0
    trainer.get_train_op(dummy_net,
                         initial_learning_rate=0.01,
                         momentum=0.9,
                         lr_decay_factor=0.1,
                         decay_steps=(1000, 2000, 3000),
                         warmup_steps=0)

