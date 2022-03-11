import sys
sys.path.append('..')
from my_python_utils.common_utils import *

import tensorflow as tf
tf.executing_eagerly()

from vtab_eval.task_adaptation.adapt_and_eval import setup_environment_gpus
setup_environment_gpus(0, memory_limit=1024)

import torchvision.models as models
import torch

import torchvision.transforms as transforms
from PIL import Image

import torch.nn as nn

from vtab_eval.port_model.port_utils import *

import tensorflow_hub as hub

def load_tf_model(handle, signature='serving_default'):
    hub_model = hub.load(
        handle, tags=None, options=None
    )
    model = hub_model.signatures[signature]
    return model

def load_image(path):
    img = cv2_imread(path)

    img = np.array(best_centercrop_image(img, 224, 224), dtype='float32') / 255.0
    return img

def load_image_tf(path):
    img = load_image(path)
    x = tf.convert_to_tensor(img.transpose((1,2,0))[None])

    return x

# add normalization, assuming input is [0,1]
class TransposeAndNormalizationLayer(nn.Module):
    def __init__(self, normalize_transform):
        super(TransposeAndNormalizationLayer, self).__init__()
        self.normalize_transform = normalize_transform

    def forward(self, x):
        # does the same as
        x = x.permute((0, 3, 1, 2))
        if self.normalize_transform is None:
            return x
        return self.normalize_transform(x)

class SqueezeLatestTwo(nn.Module):
    def __init__(self):
        super(SqueezeLatestTwo, self).__init__()

    def forward(self, x):
        return x.mean(-1).mean(-1)

class MultiplyLayer(nn.Module):
    def __init__(self, multiply_factor):
        super(MultiplyLayer, self).__init__()
        self.multiply_factor = multiply_factor

    def forward(self, x):
        return self.multiply_factor * x

# Based on
# https://gist.github.com/giangnguyen2412/ede6391ccb7b0328ca6aa14e03a0a479
def port_model(model, output_dir, normalize_transform, input_size):
    model = nn.Sequential(TransposeAndNormalizationLayer(normalize_transform),
                          model).cuda()
    model.eval()

    batch_size = 1  # random initialization

    dummy_input = torch.randn(batch_size, input_size, input_size, 3).cuda()

    input_names = ['input']
    output_names = ['output']

    dynamic_axes = {'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}}

    onnx_file = '/tmp/tmp_onnx.onnx'
    torch.onnx.export(model, dummy_input,
                      onnx_file,
                      verbose=False,
                      dynamic_axes=dynamic_axes,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=11,
                      do_constant_folding=False)

    import onnx
    from onnx_tf.backend import prepare

    model = onnx.load(onnx_file)

    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)

    # onnx.helper.printable_graph(model.graph)

    tf_rep = prepare(model, device='CPU')

    os.makedirs(output_dir, exist_ok=True)

    tf_rep.export_graph(output_dir)  # Save the model

    return tf_rep

def get_moco_resnet50(v2=True):
    import moco_training.moco.builder
    from moco_training.main_moco import parser
    # default params
    args = parser.parse_args(args=['dummy'])

    args.mlp = v2

    model = moco_training.moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)

    model = model.cuda()
    return model


def port_align_uniform(checkpoint, output_dir, remove_projection=True, test=True):
    normalize_transform = transforms.Normalize(mean=(0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
                                               std=(0.26826768628079806, 0.2610450402318512, 0.26866836876860795))

    from align_uniform.examples.stl10.encoder import VGG11, SmallAlexNet

    if 'vgg11' in checkpoint:
        torch_model = VGG11()
    elif 'small_alexnet' in checkpoint:
        torch_model = SmallAlexNet()
    torch_model.cuda()
    torch_model.load_state_dict(torch.load(checkpoint, map_location='cuda:0'))

    if remove_projection:
        blocks = list(list(torch_model.children())[0].children())
        torch_model = nn.Sequential(*blocks[:-1]).cuda()

    # add a downscaling layer to 96
    class ResizeLayer(nn.Module):
        def __init__(self, out_size):
            super(ResizeLayer, self).__init__()
            self.out_size = out_size

        def forward(self, x):
            return F.interpolate(x, scale_factor=self.out_size/224, mode='nearest')

    multiply_factor = 0.02
    torch_model = nn.Sequential(ResizeLayer(64),
                                torch_model,
                                MultiplyLayer(multiply_factor))


    torch_model.eval()
    port_model(torch_model, output_dir, normalize_transform, input_size=224)

    if test: test_model(output_dir, torch_model, normalize_transform=normalize_transform)


def port_moco_resnet50(checkpoint, output_dir, remove_projection=True, test=True):
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

    torch_model = get_moco_resnet50(True).cuda()
    torch_model.eval()
    checkpoint = torch.load(checkpoint, map_location='cuda:0')

    # rename moco_training pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = torch_model.load_state_dict(state_dict, strict=False)

    torch_model = list(list(torch_model.children())[0].children())
    if remove_projection:
        # remove projection layer
        torch_model[-1] = torch_model[-1][0]
    multiply_factor = 0.1
    torch_model = nn.Sequential(*torch_model[:-1],
                                SqueezeLatestTwo(),
                                torch_model[-1],
                                MultiplyLayer(multiply_factor)).cuda()
    torch_model.eval()
    port_model(torch_model, output_dir, normalize_transform, input_size=224)

    if test:
        test_model(output_dir, torch_model, normalize_transform=normalize_transform)

def test_easy_port_model():
    output_dir = '/tmp/tf_ported_model'

    torch_model = models.resnet50(pretrained=True).cuda()
    torch_model.eval()

    port_model(torch_model, output_dir, normalize_transform=None, input_size=224)

    test_model(output_dir, torch_model, None)

def test_model_input(tf_model, torch_model, normalize_transform, input, multiply_factor):
    tf_input = tf.convert_to_tensor(tonumpy(input))
    pytorch_input = input.permute((0, 3, 1, 2))
    if not normalize_transform is None:
        pytorch_input = normalize_transform(pytorch_input)

    try:
        output_tf = np.array(tf_model(tf_input)['output_0'])
    except:
        raise Exception("If this fails, check cuda errors on tensorflow log!")
    output_torch = tonumpy(torch_model(pytorch_input))

    top_indices_torch = np.argsort(output_torch)[0][::-1]
    top_indices_tf = np.argsort(output_tf)[0][::-1]

    # assert np.abs(top_indices_tf - top_indices_torch).sum() == 0
    assert np.allclose(output_torch, output_tf * multiply_factor, atol=1e-3)

    return output_torch


def test_model(output_dir, torch_model, normalize_transform, multiply_factor=1):
    tf_model = load_tf_model(output_dir)
    dummy_input = torch.randn(1, 224, 224, 3).cuda()
    image = torch.FloatTensor(load_image('vtab_eval/port_model/dog.jpeg')[None,...]).cuda().permute((0,2,3,1))

    output_dummy = test_model_input(tf_model, torch_model, normalize_transform, dummy_input, multiply_factor)
    output_image = test_model_input(tf_model, torch_model, normalize_transform, image, multiply_factor)

    print("MODEL: {}".format(output_dir))
    print("Output dummy: max {} min {} mean {}".format(float2str(np.abs(output_dummy).max()),
                                                       float2str(np.abs(output_dummy).min()),
                                                       float2str(np.abs(output_dummy).mean())))
    print("Output img: max {} min {} mean {}".format(float2str(np.abs(output_image).max()),
                                                     float2str(np.abs(output_image).min()),
                                                     float2str(np.abs(output_image).mean())))

    print("Output shape: {}".format(output_image.shape))

def port_supervised_pretrained_resnet50_to_tensorflow(output_dir, strip_classification=True, test=False, random_init=False):
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

    torch_model = models.resnet50(pretrained=not random_init).cuda()
    torch_model.eval()
    if strip_classification:
        torch_model = nn.Sequential(*(list(torch_model.children())[:-1]),
                                       SqueezeLatestTwo()).cuda()

    port_model(torch_model, output_dir, normalize_transform=normalize_transform, input_size=224)

    if test: test_model(output_dir, torch_model, normalize_transform)


if __name__ == '__main__':
    # port_supervised_pretrained_resnet50_to_tensorflow('vtab_eval/checkpoints_to_test/resnet50_pretrained', False, True)
    '''
    port_moco_resnet50('checkpoints_moco/pretrained_moco_v2_200/moco_v2_200ep_pretrain.pth.tar',
                       'vtab_eval/checkpoints_to_test/moco_v2_pretrained_200', remove_projection=True, test=True)
    port_moco_resnet50('checkpoints_moco/pretrained_moco_v2_800/moco_v2_800ep_pretrain.pth.tar',
                                           'vtab_eval/checkpoints_to_test/', remove_projection=True, test=True)

    #exit()
    '''

    #port_moco_resnet50('checkpoints_moco/pretrained_moco_v2_200/moco_v2_200ep_pretrain.pth.tar',
    #                    'vtab_eval/checkpoints_to_test/moco_v2_pretrained_200_no_proj', remove_projection=True, test=True)

    #moco_checkpoints = ['my_dead_leaves_mixed_large_res_256_RGB']
    #for moco_checkpoint in moco_checkpoints:
    #    port_moco_resnet50('checkpoints_moco/' + moco_checkpoint + '/checkpoint_0199.pth.tar', 'vtab_eval/checkpoints_to_test/moco_v2_' + moco_checkpoint)

    port_supervised_pretrained_resnet50_to_tensorflow('vtab_eval/checkpoints_to_test/resnet50_random', False, True, random_init=True)

    exit()
    #datasets = ['places365_RGB', 'imagenet_RGB', 'my_dead_leaves_mixed_RGB']
    datasets = ['my_dead_leaves_mixed_RGB']
    for dataset in datasets:
        for network_type in ['small_alexnet', 'vgg11']:
            port_align_uniform('results/{}/{}/align1alpha2_unif1t2_bw_True/encoder.pth'.format(dataset, network_type),
                                    'vtab_eval/checkpoints_to_test/{}_{}'.format(network_type, dataset), test=False)