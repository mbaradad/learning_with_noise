import os
import time
import argparse

import torchvision
import torch
import torch.nn as nn

from align_uniform.util import AverageMeter
from align_uniform.encoder import SmallAlexNet
from align_uniform import align_loss, uniform_loss

import logging

def parse_option():
    parser = argparse.ArgumentParser('STL-10 Representation Learning with Alignment and Uniformity Losses')

    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature')
    parser.add_argument('--align_w', type=float, default=1, help='Alignment loss weight')
    parser.add_argument('--unif_w', type=float, default=1, help='Uniformity loss weight')
    parser.add_argument('--align_alpha', type=float, default=2, help='alpha in alignment loss')
    parser.add_argument('--unif_t', type=float, default=2, help='t in uniformity loss')

    parser.add_argument('--batch_size', type=int, default=768, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default is linear scaling 0.12 per 256 batch size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', default=[155, 170, 185], nargs='*', type=int,
                        help='When to decay learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay')
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimensionality')

    parser.add_argument('--num_workers', type=int, default=20, help='Number of data loader workers to use')
    parser.add_argument('--log_interval', type=int, default=1, help='Number of iterations between logs')
    parser.add_argument('--gpus', default=[0], nargs='*', type=int,
                        help='List of GPU indices to use, e.g., --gpus 0 1 2 3')

    parser.add_argument('--result_folder', type=str, default='./result', help='Base directory to save model')

    parser.add_argument('--imagefolder', type=str, default='', help='Path to imagefolder. If not used, use STL-10.')
    parser.add_argument('--resize_image', action='store_true', help='Resize image to 96x96 (STL10 size) first')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Which optimizer to use (SGD or Adam)')

    opt = parser.parse_args()

    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    opt.gpus = list(map(lambda x: torch.device('cuda', x), opt.gpus))

    opt.save_folder = opt.result_folder
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]



def get_data_loader(opt):
    transform_array = []
    if opt.resize_image:
        transform_array.append(
            torchvision.transforms.Resize((96,96))
        )

    transform_array += [
        torchvision.transforms.RandomResizedCrop(64, scale=(0.08, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ]

    transform = torchvision.transforms.Compose(transform_array)

    transform = TwoCropsTransform(transform)

    train_path = os.path.join(opt.imagefolder, 'train')
    print(f'Loading data from {opt.imagefolder} as imagefolder')
    dataset = torchvision.datasets.ImageFolder(
        train_path,
        transform=transform)
    small_scale_samples = 105000
    assert len(dataset) == small_scale_samples, "Small scale experiment should have 105000 samples, and found {}".format(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                         shuffle=True, pin_memory=True)

    return loader


def main():
    opt = parse_option()

    torch.cuda.set_device(opt.gpus[0])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    encoder = nn.DataParallel(SmallAlexNet(feat_dim=opt.feat_dim).to(opt.gpus[0]), opt.gpus)

    if opt.optimizer == 'sgd':
        optim = torch.optim.SGD(encoder.parameters(), lr=opt.lr,
                                momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        optim = torch.optim.Adam(encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=opt.lr_decay_rate,
                                                     milestones=opt.lr_decay_epochs)

    outdir = opt.result_folder
    logfile = os.path.join(outdir, f'log_main.txt')
    # Initialize python logger
    logging.basicConfig(filename=logfile, level=logging.INFO)

    loader = get_data_loader(opt)

    loss_meter = AverageMeter('total_loss')
    it_time_meter = AverageMeter('iter_time')

    # keep two iterators to avoid waiting when starting new epoch, as the dataset is small and it matters for fast GPU's
    next_iter = loader.__iter__()
    t0 = time.time()
    for epoch in range(opt.epochs):
        actual_iter = next_iter
        next_iter = loader.__iter__()
        loss_meter.reset()
        it_time_meter.reset()
        for ii, ((im_x,im_y), _) in enumerate(actual_iter):
            optim.zero_grad()
            x, y = encoder(torch.cat([im_x.to(opt.gpus[0]), im_y.to(opt.gpus[0])])).chunk(2)

            align_loss_val = align_loss(x, y, alpha=opt.align_alpha)
            unif_loss_val = (uniform_loss(x, t=opt.unif_t) + uniform_loss(y, t=opt.unif_t)) / 2
            loss = align_loss_val * opt.align_w + unif_loss_val * opt.unif_w
            loss_meter.update(loss, x.shape[0])
            loss.backward()
            optim.step()

            it_time_meter.update(time.time() - t0)
            t0 = time.time()
            if ii % opt.log_interval == 0:
                logging_string = f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(loader)}\t" + f"{loss_meter}\t{it_time_meter}"
                logging.info(logging_string)
                print(logging_string)

        scheduler.step()

    ckpt_file = os.path.join(opt.save_folder, 'encoder.pth')
    torch.save(encoder.module.state_dict(), ckpt_file)
    print(f'Saved to {ckpt_file}')


if __name__ == '__main__':
    main()

