#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import resource

try:
    # to avoid ancdata error for too many open files, same as ulimit in console
    # maybe not necessary, but doesn't hurt
    resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))
except:
    pass

import sys
sys.path.append('.')
sys.path.append('..')

from my_python_utils.common_utils import *

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings




import dataset.webdataset.tardataset as tardataset

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils import *

import wandb

import moco_training.moco.loader
import moco_training.moco.builder
# from my_python_utils.common_utils import *
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

LARGE_SCALE_SAMPLES = 1300000

def str2bool(v):
  assert type(v) is str
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean (yes, true, t, y or 1, lower or upper case) string expected.')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', nargs='*', type=str, help='path to dataset')

parser.add_argument('--dataset_type', default='webdataset', type=str, choices=['imagefolder', 'webdataset', 'fragment_list_file'], help='path to dataset')
parser.add_argument('--n-samples', type=int, default=-1, help='n samples to use')
parser.add_argument('--rendering-gpus', type=int, default=[-1], nargs='*', help='n samples to use')

# parameters for fragment_list_file live generation
parser.add_argument('--generation-resolution-multiplier', default=1.5, type=float, help='How extra big should the images be when generating them.')

# parameters for mix-up like data generation
parser.add_argument('--mixing-type', type=str, default="convex", choices=['none', 'convex', 'cutmix', 'convex_cutmix'], help="Type of mixing strategy to use.")
parser.add_argument('--transform-before-mixing', type=str2bool, default='True', help='Whether to apply a transformation before the mixing')
parser.add_argument('--n-samples-mix', type=int, default=6)

# convex mixing parameters
parser.add_argument('--convex-combination-type', type=str, default='uniform', choices=['uniform', 'non-uniform', 'dirichlet'])
parser.add_argument('--dirichlet-alpha', type=float, default=1.0)

# cut mix mixing parameters
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--test-expected-files-found', default='False', type=str2bool, help='whether to test if the datasets have the expected number of files')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
# we use 4 gpus by default
parser.add_argument('--lr', '--learning-rate', default=0.015, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10035', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco_training v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# our custom args
parser.add_argument('--result-folder', type=str, default='./results', help='Base directory to save model')
parser.add_argument('--restart-latest', action='store_true', help='Restart latest checkpoint found in result_folder if it exists!')
parser.add_argument('--log-wandb', type=str2bool, default="True", help='Whether to use Weights and Biases login')
parser.add_argument('--restart-wandb', action='store_true', help='Restart latest checkpoint found in result_folder if it exists!')

def main():
    args = parser.parse_args()

    args.save_folder = args.result_folder

    assert args.moco_k in [65536, 1024, 4096, 16384]
    if args.moco_k != 65536:
        args.save_folder += "K_" + str(args.moco_k)
    os.makedirs(args.save_folder, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def webdataset_collate_fn(batch):
  images_to_net = []

  images_to_net.append(torch.cat([k[0][None] for k in batch[0]]))
  images_to_net.append(torch.cat([k[1][None] for k in batch[0]]))

  return images_to_net, batch[1]

global total_ii
total_ii = 0

def main_worker(gpu, ngpus_per_node, args):
    global total_ii
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        master = False
        def print_pass(*args):
            pass
        builtins.print = print_pass
    else:
        master = True

    if args.dataset_type == 'fragment_list_file':
      dataset_name = args.data[0].split('/')[-1]
    else:
      dataset_name = '_'.join([k.split('/')[-1] for k in args.data])

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print("Initializing distributed processes. If it gets stuck at this point, there is another process running/zombie with same dist_url!")
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print("End initializing distributed processes.")
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco_training.moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    # print(model)

    debug = False

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        debug = True
        model.set_debug(True)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                 momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.restart_latest:
        print("Restarting from latest checkpoint available")
        assert not args.resume, "A --resume checkpoint has been passed as argument, but restart_latest is also set (either one or the other should be set)!"
        # get latest checkpoint
        checkpoints = sorted([args.save_folder + '/' + k for k in listdir(args.save_folder, prepend_folder=False) if k.startswith('checkpoint_') and k.endswith('.pth.tar')])
        while len(checkpoints) > 0:
            checkpoint = checkpoints.pop(-1)
            # test that checkpoint was properly saved
            if checkpoint_can_be_loaded(checkpoint):
              args.resume = checkpoint
              print("Restarting from checkpoint: " + checkpoint)
              break
            else:
              print("Checkpoint {} cannot be loaded! (Probably because of an error while storing)".format(checkpoint))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if debug:
                # single gpu, remove module. from state_dict
                final_state_dict = dict()
                for k, v in checkpoint['state_dict'].items():
                    final_state_dict[k.replace('module.', '')] = v
                checkpoint['state_dict'] = final_state_dict

            args.start_epoch = checkpoint['epoch']
            total_ii = checkpoint['total_ii']
            if 'wandb_run_id' in checkpoint.keys() and not args.restart_wandb:
              wandb_run_id = checkpoint['wandb_run_id']
            elif args.log_wandb and not args.restart_wandb:
              print("Wandb loging enabled, but checkpoint does not have previous wandb_run_id!")

            if args.start_epoch >= args.epochs:
              print("All epochs already computed (desired max: {})".format(args.epochs))
              exit(0)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise Exception("=> no checkpoint found at '{}'".format(args.resume))
    else:
      print("No checkpoint resumed. Will start from scratch.")

    ''' Wandb login, extra stuff to allow resume if job is killed'''

    if master and args.log_wandb:
      exp_name = 'main-moco-{}-{}'.format(args.dataset_type, args.result_folder.split('/')[-1])
      if args.moco_k != 65536:
        exp_name += '_k_' + str(args.moco_k)
      if args.moco_t != 0.2:
        exp_name += '_moco_t_' + str(args.moco_t)

      if args.resume and not args.restart_wandb:
        resume_kwargs = dict(resume=wandb_run_id)
        print("Restarting wandb with id: " + wandb_run_id)
      else:
        wandb_run_id = exp_name + '_' + wandb.util.generate_id()
        print("Starting wandb with id: " + wandb_run_id)
        resume_kwargs = dict(id=wandb_run_id)
      wandb.init(project='noise-learning-moco',
                 name=exp_name,
                 **resume_kwargs)

      # replicate args to wandb config
      if not args.resume:
        for arg, argv in args.__dict__.items():
            wandb.config.__setattr__(arg, argv)
        if 'SLURM_JOB_ID' in os.environ.keys():
            wandb.config.__setattr__('SLURM_JOB_ID', os.environ['SLURM_JOB_ID'])

      # watch model to log parameter histograms/gradients/...
      wandb.watch(model)

    cudnn.benchmark = True

    # Data loading code
    traindirs = [os.path.join(k, 'train') for k in args.data]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_training.moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    transform = moco_training.moco.loader.TwoCropsTransform(transforms.Compose(augmentation))

    print("Loading datasets")
    train_sampler = None
    if args.dataset_type in ['fragment_list_file', 'imagefolder']:
        if args.dataset_type == 'imagefolder':
          assert len(traindirs) == 1, "Training with multiple datasets not implemented for imagefolder dataset_type."
          train_dataset = datasets.ImageFolder(
              traindirs[0],
              transform)

          if args.distributed:
              train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
          from dataset.on_the_fly_twigl.on_the_fly_twigl_fragment import TwiglOnlineDataset, get_sample_mixer

          n_samples = -1
          virtual_dataset_size = int(LARGE_SCALE_SAMPLES // args.world_size)

          fragment_list_file = args.data[0]
          fragment_files = read_text_file_lines(fragment_list_file)

          parameter_diversity = fragment_list_file.endswith('_with_parameter_diversity')
          sample_mixer = get_sample_mixer(args)

          if len(args.rendering_gpus) == 1 and args.rendering_gpus[0] == -1:
            if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
              # set the same opengl devices to those visible to cuda
              rendering_gpu = int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[gpu])
            else:
              rendering_gpu = gpu
          else:
            rendering_gpu = args.rendering_gpus

          if args.transform_before_mixing:
            transform_before_mixing = transforms.Compose([
                                        transforms.RandomResizedCrop(int(224*args.generation_resolution_multiplier), scale=(0.2, 1.)),
                                        transforms.RandomApply([
                                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                                        ], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.RandomApply([moco_training.moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                                        transforms.RandomHorizontalFlip(),
                                    ])
          else:
            transform_before_mixing = None
          print("Will use {} GPUS to render".format(rendering_gpu))
          train_dataset = TwiglOnlineDataset(fragment_files,
                                             transform,
                                             parameter_diversity=parameter_diversity,
                                             sample_mixer=sample_mixer,
                                             resolution=int(224*args.generation_resolution_multiplier),
                                             n_samples=n_samples,
                                             virtual_dataset_size=virtual_dataset_size,
                                             gpus=[rendering_gpu],
                                             max_queue_size=100000,
                                             transform_before=transform_before_mixing)

          train_sampler = None


        # set the multiprocessing context to spawn so that twigl online dataset does not get stuck:
        # https://github.com/pytorch/pytorch/issues/46409
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
            multiprocessing_context=torch.multiprocessing.get_context('fork'))
        if args.dataset_type == 'imagefolder':
          print("Training with {} samples".format(len(train_dataset)))
        else:
          print("Training with {} samples".format(len(train_dataset) * args.world_size))
    else:
        # we set epoch size to that of a single dataset (1.3M, with virtual_size parameter) so that the lr schedule per update is the same as Immgenet
        train_loader = tardataset.get_loader(traindirs,
                                             [LARGE_SCALE_SAMPLES] * len(traindirs),
                                             args.batch_size,
                                             transform=transform,
                                             num_workers=args.workers,
                                             cache_dir='',
                                             distributed=args.distributed,
                                             world_size=args.world_size,
                                             assert_n_files=args.test_expected_files_found,
                                             pin_memory=True,
                                             collate_fn=webdataset_collate_fn,
                                             virtual_size=LARGE_SCALE_SAMPLES)

    print("Datasets loaded")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and not train_sampler is None:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc1, acc5, losses = train(train_loader, model, criterion, optimizer, epoch, args, master)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

            info = {'epoch': epoch + 1,
                  'arch': args.arch,
                  'accuracy_1': acc1.avg,
                  'accuracy_5': acc5.avg,
                  'loss': losses.avg,
                  'total_ii': total_ii,
                  'wandb_run_id': wandb_run_id
                  }

            checkpoint_filename = args.save_folder + '/checkpoint_{:04d}.pth.tar'.format(epoch)
            print("Saving checkpoint in {}".format(checkpoint_filename))
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                **info
            }, is_best=False, filename=checkpoint_filename)
            # also save info as a separate file for faster loading.
            print("Finished saving checkpoint. Will save info for faster loading!")
            save_checkpoint(info, is_best=False, filename=args.save_folder + '/info_{:04d}.pth.tar'.format(epoch))
            print("Finished saving!")
    print("Finished training!")
    exit(1)

def train(train_loader, model, criterion, optimizer, epoch, args, master):
    global total_ii
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    print("Starting epoch:")

    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        cur_data_time = time.time() - end
        data_time.update(cur_data_time)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)


        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        cur_batch_time = time.time() - end
        batch_time.update(cur_batch_time)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            if args.log_wandb and master:
                wandb.log({'train_loss/loss': float(loss.item()),
                           'train_loss/top1': float(acc1[0]),
                           'train_loss/top5': float(acc5[0]),
                           'lr/lr': optimizer.state_dict()['param_groups'][0]['lr'],
                           'iter_num': total_ii,
                           'times/batch_time': cur_batch_time,
                           'times/data_time': cur_data_time})

        if master:
          total_ii += 1

    return top1, top5, losses

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
