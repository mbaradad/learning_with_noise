import torch
from torch import nn

from tqdm import trange

import numpy as np
import time

import os,sys
assert os.getcwd().endswith('generate_datasets'), "This script should be run from generate_datasets folder."

npath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stylegan2-ada-pytorch')
if not npath in sys.path:
    sys.path.append(npath)
npath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stylegan2-ada-pytorch/training')
if not npath in sys.path:
    sys.path.append(npath)

import networks as networks_original # for default
import network_randomized as networks_random # for randomness, no sparsity
import network_randomized_v2 as networks_sparse # for sparsity
import network_randomized_v3 as networks_sparse_new # for better wavelet sampling

sys.path.append('.')

from utils_generation import *

from PIL import Image

def tensor_to_uint8(x, nrm=None):
    if x.ndim==4:
        x = x[0]

    x = x.detach().cpu().numpy()
    x = x.transpose(1,2,0)
    mx = max(np.abs(x.max()), np.abs(x.min()))

    mx = np.percentile(np.abs(x),95)

    if nrm is None:
        xmn = x.min()
        xmx = x.max()
    else:
        xmn,xmx = nrm

        xmn = min(xmn,-mx)
        xmx = max(xmx,mx)
    x = (x-xmn) / (xmx - xmn) * 255
    x = np.clip(x,0,255).astype('uint8')
    return x


# New image normalization
def tensor_to_uint8_norm(x):
    if x.ndim==4:
        return np.array([tensor_to_uint8(x_) for x_ in x])

    x = x.detach().cpu().numpy()
    # normalize each channel separately

    ch1_mean = np.random.randn()* 0.145 + 0.483
    ch1_std = np.random.randn() * 0.063 + 0.219
    ch2_mean = np.random.randn() * 0.142 + 0.455
    ch2_std = np.random.randn() * 0.062 + 0.213
    ch3_mean = np.random.randn() * 0.161 + 0.401
    ch3_std = np.random.randn() * 0.069 + 0.213

    meanstd = [(ch1_mean, ch1_std),
            (ch2_mean, ch2_std),
            (ch3_mean, ch3_std)]

    # normalize x
    for i in range(3):
        x[i] = (x[i]-x[i].mean()) / x[i].std()

        x[i] = x[i]*meanstd[i][1] + meanstd[i][0]

    x = x.transpose(1,2,0)

    x = (np.clip(x, 0, 1) * 255).astype('uint8')
    return x


R = nn.LeakyReLU(0.2)


def get_network(args):
    if args.network_type == 'original':
        nets = networks_original
        synthesis_kwargs={
            'activation':'relu',
            # 'resample_filter': [1,2,1],
        }

    elif args.network_type == 'random':
        nets = networks_random
        synthesis_kwargs={
            'activation':'relu',
            # 'resample_filter': [1,2,1],
        }

    elif args.network_type == 'sparse':
        nets = networks_sparse
        synthesis_kwargs={
            'activation':'relu',
            # 'resample_filter': [1,2,1],
            'same_noise_maps': args.same_noise_map,
            'bias_range': args.bias_range
        }

    elif args.network_type == 'sparse_new':
        nets = networks_sparse_new
        synthesis_kwargs={
            'activation':'relu',
            # 'resample_filter': [1,2,1],
            'same_noise_maps': args.same_noise_map,
            'bias_range': args.bias_range
        }



    G = nets.Generator(z_dim=512,
                       c_dim=0,
                       w_dim=512,
                       img_resolution=args.res,
                       img_channels=3,
                       synthesis_kwargs=synthesis_kwargs)
    G.eval()

    nl = 14
    return G.cuda(),nl

def generate_images(G,nimg,args):
    nl = G.num_ws

    out = []
    w = torch.randn(nimg,nl,512).cuda()
    w_ = R(w)

    if args.network_type == 'original':
        noise_mode='random'
    else:
        noise_mode = 'pink_random'

    if args.network_type == 'sparse_new':
        if args.random_configuration == 'chin-chout':
            rand_cfg = ('chin','chout')
        elif args.random_configuration == 'chout-chin':
            rand_cfg = ('chout', 'chin')
        else:
            raise "--random_configuration {} not implemented!".format(args.random_configuration)

        I = G.synthesis(w_, noise_mode=noise_mode, random_configuration=rand_cfg)
    else:
        I = G.synthesis(w_, noise_mode=noise_mode)

    if args.old_color_norm:
        out = [tensor_to_uint8(i, nrm=(-5,5)) for i in I]
    else:
        out = [tensor_to_uint8_norm(i) for i in I]
    return out

def large_scale_options(opt):
    # both small and large scale are at 256
    # opt.resolution = 256
    opt.nimg = 1300000
    return opt

def small_scale_options(opt):
    # both small and large scale are at 256
    # opt.resolution = 128
    opt.nimg = 105000
    return opt

def get_output_path(opt):
    if opt.name == '':
        outpath = '../generated_datasets/{}/stylegan-'.format('small_scale' if opt.small_scale else 'large_scale')
        if opt.network_type == 'original':
            outpath += 'random'
        elif opt.network_type == 'random':
            outpath += 'highfreq'
#        elif opt.network_type == 'sparse' and opt.bias_range == 0.2:
#            outpath += 'sparse'
        elif opt.network_type == 'sparse_new' and opt.bias_range == 0.2 and opt.random_configuration == 'chin-chout':
            outpath += 'sparse'
        elif opt.network_type == 'sparse_new' and opt.bias_range == 0.2 and opt.random_configuration == 'chin-chout' and opt.same_noise_map:
            outpath += 'oriented'
        else:
            raise Exception("No --name provided and is not one of the datsets in the paper.")
    else:
        if opt.name.startswith('/'):
            outpath = opt.name
        else:
            outpath = f'./generated-datasets/{opt.name}'

    outpath = os.path.join(outpath, 'train')

    return outpath

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--small-scale', action="store_true", help="use small-scale default parameters")
    parser.add_argument('--large-scale', action="store_true", help="use large-scale default parameters")

    parser.add_argument('--res', type=int, default=256, choices=[256,512,1024])
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--nimg', type=int, default=100000)
    parser.add_argument('--startimg', type=int, default=0)
    parser.add_argument('--same_noise_map', action='store_true')
    parser.add_argument('--bias_range', type=float, default=0.0)
    parser.add_argument('--network_type', type=str, required=True, choices=['original', 'random', 'sparse', 'sparse_new'])
    parser.add_argument('--old_color_norm', action='store_true', help='Use old color normalization')
    parser.add_argument('--random_configuration', type=str, default='chin-chout')
    parser.add_argument('--name', type=str, default='',  help='Name of dataset. If empty, use default.')

    opt = parser.parse_args()

    select_gpus(opt.gpu)

    if opt.large_scale:
        opt = large_scale_options(opt)
    elif opt.small_scale:
        opt = small_scale_options(opt)

    outpath = get_output_path(opt)

    BS = 2

    with torch.no_grad():
        for i in trange(opt.nimg//BS):
            imgnum = i*BS + opt.startimg
            topnum = imgnum // 1000
            outdir = os.path.join(
                outpath, f'{topnum:08d}'
            )
            os.makedirs(outdir, exist_ok=True)

            # Reset every 100 batches
            if i % (400 / BS) == 0:
                G,nl = get_network(opt)

            images = generate_images(G,BS,opt)

            for j in range(BS):
                inum = imgnum + j
                fname = os.path.join(outdir, f'{inum:08d}.jpg')
                Image.fromarray(images[j]).save(fname, quality=95)


if __name__ == '__main__':
    main()
