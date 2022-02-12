# Copyright 2020 The Lucent Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import, division, print_function

from my_python_utils.common_utils import *

import warnings
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

from lucent.optvis import transform, param
from lucent.misc.io import show

import vis.grad_ascent.lucent.objectives as objectives

def render_vis(
    model,
    objective_f,
    param_f=None,
    optimizer=None,
    transforms=None,
    thresholds=(512,),
    verbose=False,
    preprocess=True,
    progress=True,
    show_image=True,
    save_image=False,
    image_name=None,
    show_inline=False,
    fixed_image_size=None,
    pool_mode='mean',
    negative_activation=False,
    break_on_error=False,
    learning_rate=5e-2,
    debug=False,
    fft=True,
    std_init=0.01
):
    assert pool_mode in ['mean', 'single']

    if type(objective_f) is str:
        objective_f = [objective_f]

    if negative_activation and type(objective_f) is str:
        # assert it is not a relu
        assert not 'relu' in objective_f.lower(), "Layer has to be a conv when using negatie activation, and it is: {}".format(objective_f)

    batch_size = len(objective_f)
    if param_f is None:
        param_f = lambda: param.image(fixed_image_size, batch=batch_size, fft=fft, sd=std_init)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()

    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=learning_rate)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = transform.standard_transforms.copy()

    if preprocess:
        if model._get_name() == "InceptionV1":
            # Original Tensorflow InceptionV1 takes input range [-117, 138]
            transforms.append(transform.preprocess_inceptionv1())
        else:
            # Assume we use normalization for torchvision.final_models
            # See https://pytorch.org/docs/stable/torchvision/models.html
            transforms.append(transform.normalize())

    # Upsample images smaller than 224
    image_shape = image_f().shape
    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    hook = hook_model(model, image_f)
    objective_f, layer, chns = objectives.as_objective(objective_f, pool_mode=pool_mode, negative_activation=negative_activation)

    if verbose:
        model(transform_f(image_f()))
        print("Initial loss: {:.3f}".format(objective_f(hook)))

    images = []
    loss_values = []
    try:
        for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
            optimizer.zero_grad()
            try:
                model(transform_f(image_f()))
            except RuntimeError as ex:
                if i == 1:
                    # Only display the warning message
                    # on the first iteration, no need to do that
                    # every iteration
                    warnings.warn(
                        "Some layers could not be computed because the size of the "
                        "image is not big enough. It is fine, as long as the non"
                        "computed layers are not used in the objective function"
                        f"(exception details: '{ex}')"
                    )
                    print(ex)
                if break_on_error: raise ex
            losses = objective_f(hook)
            loss = losses.mean()
            loss.backward()
            optimizer.step()

            actual_loss_values = losses.detach().cpu().numpy()
            if i < 10:
                # reinit images for which the losses is 0, as it is probably saturating a relu:
                failed_init = (actual_loss_values == 0).sum()

            loss_values.append(actual_loss_values)

            if i in thresholds:
                image = tensor_to_img_array(image_f())
                if verbose:
                    print("Loss at step {}: {:.3f}".format(i, loss))
                    if show_inline:
                        show(image)
                images.append(image)
    except KeyboardInterrupt as ex:
        print("Interrupted optimization at step {:d}.".format(i))
        if verbose:
            print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
        images.append(tensor_to_img_array(image_f()))
        if break_on_error: raise ex

    if save_image:
        export(image_f(), image_name)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())
    return images, loss_values


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def export(tensor, image_name=None):
    image_name = image_name or "image.jpg"
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).save(image_name)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        self.hook.remove()

def get_layers_by_layer_name(model):
    layers = OrderedDict()
    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                layers["_".join(prefix + [name])] = layer
                get_layers(layer, prefix=prefix + [name])
    get_layers(model)
    return layers

def hook_model(model, image_f):
    features = OrderedDict()

    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    return hook
