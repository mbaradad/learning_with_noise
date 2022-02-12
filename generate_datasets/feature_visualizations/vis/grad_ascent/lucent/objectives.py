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

import numpy as np
import torch
import torch.nn.functional as F
from decorator import decorator
from lucent.optvis.objectives_util import _make_arg_str, _extract_act_pos, _T_handle_batch

from my_python_utils.common_utils import *

class Objective():

    def __init__(self, objective_func, name="", description=""):
        self.objective_func = objective_func
        self.name = name
        self.description = description

    def __call__(self, model):
        return self.objective_func(model)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda model: other + self(model)
            name = self.name
            description = self.description
        else:
            objective_func = lambda model: self(model) + other(model)
            name = ", ".join([self.name, other.name])
            description = "Sum(" + " +\n".join([self.description, other.description]) + ")"
        return Objective(objective_func, name=name, description=description)

    @staticmethod
    def sum(objs):
        objective_func = lambda T: sum([obj(T) for obj in objs])
        descriptions = [obj.description for obj in objs]
        description = "Sum(" + " +\n".join(descriptions) + ")"
        names = [obj.name for obj in objs]
        name = ", ".join(names)
        return Objective(objective_func, name=name, description=description)

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-1 * other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda model: other * self(model)
            return Objective(objective_func, name=self.name, description=self.description)
        else:
            # Note: In original Lucid library, objectives can be multiplied with non-numbers
            # Removing for now until we find a good use case
            raise TypeError('Can only multiply by int or float. Received type ' + str(type(other)))

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(1 / other)
        else:
            raise TypeError('Can only divide by int or float. Received type ' + str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)


def wrap_objective():
    @decorator
    def inner(func, *args, **kwds):
        objective_func = func(*args, **kwds)
        objective_name = func.__name__
        args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
        description = objective_name.title() + args_str
        return Objective(objective_func, objective_name, description)
    return inner


def handle_batch(batch=None):
    return lambda f: lambda model: f(_T_handle_batch(model, batch=batch))


@wrap_objective()
def neuron(layer, n_channel, x=None, y=None, batch=None):
    """Visualize a single neuron of a single channel.

    Defaults to the center neuron. When width and height are even numbers, we
    choose the neuron in the bottom right of the center 2x2 neurons.

    Odd width & height:               Even width & height:

    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   | X |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   | X |   |
    +---+---+---+                     +---+---+---+---+
                                      |   |   |   |   |
                                      +---+---+---+---+

    """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        return -layer_t[:, n_channel].mean()
    return inner


class ChannelStuff():
    def __init__(self, channels):
        if channels[0] == 'random':
            assert all([k == 'random' for k in
                        channels]), "Different mode not implemented, all must be random or fixed channel"
            self.random_mode = True
        else:
            self.random_mode = False
        self.channels = channels

        self.max_channel = None
        self.mask = None
        self.mask_sum = None
        self.is_instantiated = False

    def instiantate_mask(self, activation_shape, pool_mode):
        if not self.random_mode:
            max_channel = max([max(k) for k in self.channels]) + 1
            if len(activation_shape) == 2:
                mask = torch.zeros((len(self.channels), max_channel)).cuda()
            if len(activation_shape) == 3:
                mask = torch.zeros((len(self.channels), max_channel, activation_shape[2])).cuda()
            elif len(activation_shape) == 4:
                mask = torch.zeros((len(self.channels), max_channel, activation_shape[2], activation_shape[3])).cuda()
            for i, js in enumerate(self.channels):
                for j in js:
                    if pool_mode == 'mean':
                        mask[i,j] = 1
                    elif pool_mode == 'single':
                        if len(activation_shape) == 2:
                            mask[i,j] = 1
                        elif len(activation_shape) == 3:
                            mask[i, j, activation_shape[2]//2 ] = 1
                        elif len(activation_shape) == 4:
                            mask[i, j, activation_shape[2] // 2, activation_shape[3] // 2] = 1
                    else:
                        raise Exception("Pool mode {} not implemented!".format(pool_mode))
            mask_sum = mask.sum()
        else:
            max_channel = activation_shape[1]
            mask = torch.FloatTensor(np.random.normal(size=activation_shape)).cuda()
            mask_sum = mask.nelement()

        self.max_channel = max_channel
        self.mask_sum = mask_sum
        self.mask = mask
        self.is_instantiated = True

    def get_stuff(self):
        return self.max_channel, self.mask, self.mask_sum

@wrap_objective()
def channel(layer, channels, pool_mode='mean', batch=None, negative_activation=False):
    """Visualize a single channel"""

    channel_stuff = ChannelStuff(channels)

    @handle_batch(batch)
    def inner(model):
        layer_activation = model(layer)
        if not channel_stuff.is_instantiated:
            channel_stuff.instiantate_mask(layer_activation.shape, pool_mode)
        max_channel, mask, mask_sum = channel_stuff.get_stuff()
        vals = -layer_activation[:,:max_channel] * mask
        if len(vals.shape) == 4:
            vals = vals.sum(-1).sum(-1).sum(-1) / mask_sum
        if len(vals.shape) == 3:
            vals = vals.sum(-1).sum(-1) / mask_sum
        if len(vals.shape) == 2:
            vals = vals.sum(-1) / mask_sum
        assert len(vals.shape) == 1, "Check objective, at this point there should be size shape!"
        if negative_activation:
            vals = -vals
        return vals
    return inner

@wrap_objective()
def neuron_weight(layer, weight, x=None, y=None, batch=None):
    """ Linearly weighted channel activation at one location as objective
    weight: a torch Tensor vector same length as channel.
    """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        if weight is None:
            return -layer_t.mean()
        else:
            return -(layer_t.squeeze() * weight).mean()
    return inner

@wrap_objective()
def channel_weight(layer, weight, batch=None):
    """ Linearly weighted channel activation as objective
    weight: a torch Tensor vector same length as channel. """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        return -(layer_t * weight.view(1, -1, 1, 1)).mean()
    return inner

@wrap_objective()
def localgroup_weight(layer, weight=None, x=None, y=None, wx=1, wy=1, batch=None):
    """ Linearly weighted channel activation around some spot as objective
    weight: a torch Tensor vector same length as channel. """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        if weight is None:
            return -(layer_t[:, :, y:y + wy, x:x + wx]).mean()
        else:
            return -(layer_t[:, :, y:y + wy, x:x + wx] * weight.view(1, -1, 1, 1)).mean()
    return inner

@wrap_objective()
def direction(layer, direction, batch=None):
    """Visualize a direction

    InceptionV1 example:
    > direction = torch.rand(512, device=device)
    > obj = objectives.direction(layer='mixed4c', direction=direction)

    Args:
        layer: Name of layer in model (string)
        direction: Direction to visualize. torch.Tensor of shape (num_channels,)
        batch: Batch number (int)

    Returns:
        Objective

    """

    @handle_batch(batch)
    def inner(model):
        return -torch.nn.CosineSimilarity(dim=1)(direction.reshape(
            (1, -1, 1, 1)), model(layer)).mean()

    return inner


@wrap_objective()
def direction_neuron(layer,
                     direction,
                     x=None,
                     y=None,
                     batch=None):
    """Visualize a single (x, y) position along the given direction

    Similar to the neuron objective, defaults to the center neuron.

    InceptionV1 example:
    > direction = torch.rand(512, device=device)
    > obj = objectives.direction_neuron(layer='mixed4c', direction=direction)

    Args:
        layer: Name of layer in model (string)
        direction: Direction to visualize. torch.Tensor of shape (num_channels,)
        batch: Batch number (int)

    Returns:
        Objective

    """

    @handle_batch(batch)
    def inner(model):
        # breakpoint()
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        return -torch.nn.CosineSimilarity(dim=1)(direction.reshape(
            (1, -1, 1, 1)), layer_t).mean()

    return inner


def _torch_blur(tensor, out_c=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    depth = tensor.shape[1]
    weight = np.zeros([depth, depth, out_c, out_c])
    for ch in range(depth):
        weight_ch = weight[ch, ch, :, :]
        weight_ch[ :  ,  :  ] = 0.5
        weight_ch[1:-1, 1:-1] = 1.0
    weight_t = torch.tensor(weight).float().to(device)
    conv_f = lambda t: F.conv2d(t, weight_t, None, 1, 1)
    return conv_f(tensor) / conv_f(torch.ones_like(tensor))


@wrap_objective()
def blur_input_each_step():
    """Minimizing this objective is equivelant to blurring input each step.
    Optimizing (-k)*blur_input_each_step() is equivelant to:
    input <- (1-k)*input + k*blur(input)
    An operation that was used in early feature visualization work.
    See Nguyen, et al., 2015.
    """
    def inner(T):
        t_input = T("input")
        with torch.no_grad():
            t_input_blurred = _torch_blur(t_input)
        return -0.5*torch.sum((t_input - t_input_blurred)**2)
    return inner


@wrap_objective()
def channel_interpolate(layer1, n_channel1, layer2, n_channel2):
    """Interpolate between layer1, n_channel1 and layer2, n_channel2.
    Optimize for a convex combination of layer1, n_channel1 and
    layer2, n_channel2, transitioning across the batch.
    Args:
        layer1: layer to optimize 100% at batch=0.
        n_channel1: neuron index to optimize 100% at batch=0.
        layer2: layer to optimize 100% at batch=N.
        n_channel2: neuron index to optimize 100% at batch=N.
    Returns:
        Objective
    """
    def inner(model):
        batch_n = list(model(layer1).shape)[0]
        arr1 = model(layer1)[:, n_channel1]
        arr2 = model(layer2)[:, n_channel2]
        weights = np.arange(batch_n) / (batch_n - 1)
        sum_loss = 0
        for n in range(batch_n):
            sum_loss -= (1 - weights[n]) * arr1[n].mean()
            sum_loss -= weights[n] * arr2[n].mean()
        return sum_loss
    return inner


@wrap_objective()
def alignment(layer, decay_ratio=2):
    """Encourage neighboring images to be similar.
    When visualizing the interpolation between two objectives, it's often
    desirable to encourage analogous objects to be drawn in the same position,
    to make them more comparable.
    This term penalizes L2 distance between neighboring images, as evaluated at
    layer.
    In general, we find this most effective if used with a parameterization that
    shares across the batch. (In fact, that works quite well by itself, so this
    function may just be obsolete.)
    Args:
        layer: layer to penalize at.
        decay_ratio: how much to decay penalty as images move apart in batch.
    Returns:
        Objective.
    """
    def inner(model):
        batch_n = list(model(layer).shape)[0]
        layer_t = model(layer)
        accum = 0
        for d in [1, 2, 3, 4]:
            for i in range(batch_n - d):
                a, b = i, i + d
                arr_a, arr_b = layer_t[a], layer_t[b]
                accum += ((arr_a - arr_b) ** 2).mean() / decay_ratio ** float(d)
        return accum
    return inner


@wrap_objective()
def diversity(layer):
    """Encourage diversity between each batch element.

    A neural net feature often responds to multiple things, but naive feature
    visualization often only shows us one. If you optimize a batch of images,
    this objective will encourage them all to be different.

    In particular, it calculates the correlation matrix of activations at layer
    for each image, and then penalizes cosine similarity between them. This is
    very similar to ideas in style transfer, except we're *penalizing* style
    similarity instead of encouraging it.

    Args:
        layer: layer to evaluate activation correlations on.

    Returns:
        Objective.
    """
    def inner(model):
        layer_t = model(layer)
        batch, channels, _, _ = layer_t.shape
        flattened = layer_t.view(batch, channels, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = F.normalize(grams, p=2, dim=(1, 2))
        return -sum([ sum([ (grams[i]*grams[j]).sum()
               for j in range(batch) if j != i])
               for i in range(batch)]) / batch
    return inner


def as_objective(obj, pool_mode='mean', negative_activation=False):
    """Convert obj into Objective class.

    Strings of the form "layer:n" become the Objective channel(layer, n).
    Objectives are returned unchanged.

    Args:
        obj: string or Objective.

    Returns:
        Objective
    """

    assert pool_mode in ['mean', 'single']

    if isinstance(obj, Objective):
        return obj
    if callable(obj):
        return obj
    if isinstance(obj, str):
        obj = [obj]
    if isinstance(obj, list):
        elems = [elem.split(":") for elem in obj]
        layers = [k[0] for k in elems]
        channels = []
        for k in elems:
            channel_string = k[1]
            if channel_string == 'random':
                channels.append('random')
            else:
                channels.append([int(j) for j in channel_string.split('_')])
        assert len(set(layers)) == 1, "Multibatch only implemented for filters on the same layer and more than one specified ({})".format(layers)
        layer = layers[0]
        return channel(layer, channels, pool_mode=pool_mode, negative_activation=negative_activation), layer, channels
