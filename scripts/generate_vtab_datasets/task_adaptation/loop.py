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

"""Main adaptation and evaluation loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags

import vtab_eval.task_adaptation.data_loader as data_loader
import vtab_eval.task_adaptation.model as model

import tensorflow.compat.v1 as tf
import numpy as np

# TPU-specific constant, see
# https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUConfig for more
# details.
TPU_ITERATION_PER_LOOP = 300

FLAGS = flags.FLAGS


def setup_estimator(
    hub_module,
    hub_module_signature,
    work_dir,
    tpu_name,
    save_checkpoints_steps,
    optimization_params,
    data_params):
  """Produces TPUEstimator object for a given configuration."""

  # Merge all parameters into single dictionary (for tf.estimator API).
  num_classes = data_params["dataset"].get_num_classes()
  params = {k: v for d in [optimization_params, data_params,
                           {"hub_module": hub_module,
                            "hub_module_signature": hub_module_signature,
                            "num_classes": num_classes}]
            for k, v in d.items()}

  # Defines the configutation of an adaptation/evaluation loop.

  if tpu_name is not None:
    cluster = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    config = tf.contrib.tpu.RunConfig(
        model_dir=work_dir,
        cluster=cluster,
        keep_checkpoint_max=None,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=TPU_ITERATION_PER_LOOP))
  else:
    config = tf.estimator.RunConfig(
        model_dir=work_dir,
        keep_checkpoint_max=None,
        save_checkpoints_steps=save_checkpoints_steps,
        log_step_count_steps=2)

  if tpu_name is not None:
    batch_size = params.pop("batch_size")
    batch_size_eval = params.pop("batch_size_eval")
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model.model_fn,
        model_dir=work_dir,
        params=params,
        config=config,
        use_tpu=True,
        train_batch_size=batch_size,
        eval_batch_size=batch_size_eval)
  else:
    params.pop('data_dir')
    estimator = MyEstimator(
        debug=optimization_params['debug'],
        model_fn=model.model_fn,
        model_dir=work_dir,
        params=params,
        config=config)

  return estimator

from tensorflow.python.summary import summary
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import run_config
from tensorflow.python.training import training
# copy pasted from tf.estimator.Estimator, to debug
class MyEstimator(tf.estimator.Estimator):
    def __init__(self, debug=False, *args, **kwargs):
        tf.estimator.Estimator._assert_members_are_not_overridden = lambda self: None
        super(MyEstimator, self).__init__(*args, **kwargs)
        self.debug = debug

    def _train_with_estimator_spec(self, estimator_spec, worker_hooks, hooks,
                                   global_step_tensor, saving_listeners):
        """Train a model with the given Estimator Spec."""
        if (self._warm_start_settings and
                not tf.train.latest_checkpoint(self._model_dir)):
            tf.compat.v1.logging.info('Warm-starting with WarmStartSettings: %s' %
                                      (self._warm_start_settings,))
            tf.compat.v1.train.warm_start(*self._warm_start_settings)
        # Check if the user created a loss summary, and add one if they didn't.
        # We assume here that the summary is called 'loss'. If it is not, we will
        # make another one with the name 'loss' to ensure it shows up in the right
        # graph in TensorBoard.
        if not any([
            x.op.name == 'loss' for x in ops.get_collection(ops.GraphKeys.SUMMARIES)
        ]):
            summary.scalar('loss', estimator_spec.loss)
        ops.add_to_collection(ops.GraphKeys.LOSSES, estimator_spec.loss)
        worker_hooks.extend(hooks)
        worker_hooks.append(tf.compat.v1.train.NanTensorHook(estimator_spec.loss))
        if self._config.log_step_count_steps is not None:
            worker_hooks.append(
                tf.compat.v1.train.LoggingTensorHook(
                    {
                        'loss': estimator_spec.loss,
                        'step': global_step_tensor
                    },
                    every_n_iter=self._config.log_step_count_steps))
        worker_hooks.extend(estimator_spec.training_hooks)

        if not (estimator_spec.scaffold.saver or
                tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SAVERS)):
            tf.compat.v1.add_to_collection(
                tf.compat.v1.GraphKeys.SAVERS,
                tf.compat.v1.train.Saver(
                    sharded=True,
                    max_to_keep=self._config.keep_checkpoint_max,
                    keep_checkpoint_every_n_hours=(
                        self._config.keep_checkpoint_every_n_hours),
                    defer_build=True,
                    save_relative_paths=True))

        if (self._config.cluster_spec and type(
                self._train_distribution).__name__ in ('CollectiveAllReduceStrategy',
                                                       'CollectiveAllReduceStrategyV1',
                                                       'MultiWorkerMirroredStrategy')):
            return self._train_with_estimator_spec_distributed(
                estimator_spec, worker_hooks, saving_listeners)

        chief_hooks = []
        all_hooks = worker_hooks + list(estimator_spec.training_chief_hooks)
        saver_hooks = [
            h for h in all_hooks
            if isinstance(h, tf.compat.v1.train.CheckpointSaverHook)
        ]
        if (self._config.save_checkpoints_secs or
                self._config.save_checkpoints_steps):
            if not saver_hooks:
                chief_hooks = [
                    tf.compat.v1.train.CheckpointSaverHook(
                        self._model_dir,
                        save_secs=self._config.save_checkpoints_secs,
                        save_steps=self._config.save_checkpoints_steps,
                        scaffold=estimator_spec.scaffold)
                ]
                saver_hooks = [chief_hooks[0]]
        if saving_listeners:
            if not saver_hooks:
                raise ValueError(
                    'There should be a CheckpointSaverHook to use saving_listeners. '
                    'Please set one of the RunConfig.save_checkpoints_steps or '
                    'RunConfig.save_checkpoints_secs.')
            else:
                # It is expected to have one CheckpointSaverHook. If multiple, we pick
                # up the first one to add listener.
                for listener in saving_listeners:
                    # pylint: disable=protected-access
                    if listener not in saver_hooks[0]._listeners:
                        saver_hooks[0]._listeners.append(listener)
                    # pylint: disable=protected-access

        # Add summary hooks to worker 0 if we are running with a master, to ensure
        # that summaries are written at correct intervals even with long-running
        # evaluations.
        save_summary_steps = self._config.save_summary_steps
        log_step_count_steps = self._config.log_step_count_steps

        # Check existence of appropriate cluster spec fields, as well as master and
        # worker nodes. As master also performs evaluation, summary writing must
        # occur on a different node. The presence of a worker is also checked to
        # prevent reassigning hooks for single-replica jobs with just a master node.
        if (self._config.cluster_spec and self._config.cluster_spec.jobs and
                (run_config.TaskType.WORKER in self._config.cluster_spec.jobs) and
                (run_config.TaskType.MASTER in self._config.cluster_spec.jobs)):
            # Update config values to prevent the default hooks from being created on
            # the master or other workers.
            save_summary_steps = 0
            log_step_count_steps = None

            if (self._config.task_type == run_config.TaskType.WORKER and
                    self._config.task_id == 0):
                if (self._config.save_summary_steps and
                        self._config.save_summary_steps > 0):
                    worker_hooks.append(
                        tf.compat.v1.train.SummarySaverHook(
                            save_steps=self._config.save_summary_steps,
                            output_dir=self._config.model_dir,
                            scaffold=estimator_spec.scaffold))

                if (self._config.log_step_count_steps and
                        self._config.log_step_count_steps > 0):
                    worker_hooks.append(
                        tf.compat.v1.train.StepCounterHook(
                            every_n_steps=self._config.log_step_count_steps,
                            output_dir=self._config.model_dir))

        with training.MonitoredTrainingSession(
                master=self._config.master,
                is_chief=self._config.is_chief,
                checkpoint_dir=self._model_dir,
                scaffold=estimator_spec.scaffold,
                hooks=worker_hooks,
                chief_only_hooks=(tuple(chief_hooks) +
                                  tuple(estimator_spec.training_chief_hooks)),
                save_checkpoint_secs=0,  # Saving is handled by a hook.
                save_summaries_steps=save_summary_steps,
                config=self._session_config,
                max_wait_secs=self._config.session_creation_timeout_secs,
                log_step_count_steps=log_step_count_steps) as mon_sess:
            loss = None
            any_step_done = False
            while not mon_sess.should_stop():
                if not estimator_spec.predictions is None and self.debug:
                    _, loss, predictions = mon_sess.run([estimator_spec.train_op, estimator_spec.loss, estimator_spec.predictions])
                    print("Loss v: {}".format(loss))
                    for k, v in predictions.items():
                        v = np.abs(v)
                        if 'gradient' in k:
                            print("Gradient {}: max {:.2f} min {:.2f} mean {:.2f}".format(k, v.max(), v.min(), v.mean()))
                        if 'embedding' in k:
                            print("Embedding {}: max {:.2f} min {:.2f} mean {:.2f}".format(k, v.max(), v.min(), v.mean()))
                else:
                    _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
                any_step_done = True
        if not any_step_done:
            tf.compat.v1.logging.warn('Training with estimator made no steps. '
                                      'Perhaps input is empty or misspecified.')
        return loss


def run_training_loop(hub_module,
                      hub_module_signature,
                      work_dir,
                      tpu_name,
                      save_checkpoints_steps,
                      optimization_params,
                      data_params):
  """Runs training loop."""
  data_params["dataset"] = data_loader.get_dataset_instance(data_params)
  estimator = setup_estimator(hub_module,
                              hub_module_signature,
                              work_dir,
                              tpu_name,
                              save_checkpoints_steps,
                              optimization_params,
                              data_params)
  input_fn = data_loader.build_data_pipeline(data_params, mode="train")


  previous_checkpoint_exists = False
  latest_checkpoint = tf.train.latest_checkpoint(estimator._model_dir)
  if not latest_checkpoint is None:
      checkpoint_reader = tf.compat.v1.train.NewCheckpointReader(
          tf.train.latest_checkpoint(estimator._model_dir))
      previous_steps = checkpoint_reader.get_tensor(tf.compat.v1.GraphKeys.GLOBAL_STEP)
      previous_checkpoint_exists = previous_steps == optimization_params["max_steps"]

  # TPUs require the max number of steps to be specified explicitly.
  estimator.train(input_fn, max_steps=optimization_params["max_steps"])

  # if train_linear_only, assert nothing else only linear layers have changed



def run_evaluation_loop(hub_module,
                        hub_module_signature,
                        work_dir,
                        tpu_name,
                        save_checkpoints_steps,
                        optimization_params,
                        data_params):
  """Runs evaluation loop."""
  data_params["dataset"] = data_loader.get_dataset_instance(data_params)
  estimator = setup_estimator(hub_module,
                              hub_module_signature,
                              work_dir,
                              tpu_name,
                              save_checkpoints_steps,
                              optimization_params,
                              data_params)
  input_fn = data_loader.build_data_pipeline(data_params, mode="eval")

  with tf.gfile.Open(os.path.join(work_dir, "result_file.txt"), "w") as f:
    all_checkpoints = set([".".join(f.split(".")[:-1])
                           for f in tf.gfile.ListDirectory(work_dir)
                           if f.startswith("model.ckpt")])
    # Sort checkpoints by the global step.
    all_checkpoints = sorted(all_checkpoints,
                             key=lambda x: int(x.split("-")[-1]))
    # For efficiency reasons we evluate only the last checkpoint
    for ckpt in all_checkpoints[-1:]:
      ckpt = os.path.join(work_dir, ckpt)
      res = estimator.evaluate(input_fn,
                               steps=(data_params["dataset"].get_num_samples(
                                   data_params["dataset_eval_split_name"]) //
                                      data_params["batch_size_eval"]),
                               checkpoint_path=ckpt)
      f.write("Accuracy at step {}: {}\n".format(res["global_step"],
                                                 res["accuracy"]))
