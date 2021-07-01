import time
from datetime import datetime # to log file naming
import argparse

import sys,os
import logging

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from align_uniform.util import AverageMeter
from align_uniform.encoder import SmallAlexNet


def parse_option():
    parser = argparse.ArgumentParser('STL-10 Representation Learning with Alignment and Uniformity Losses')
    parser.add_argument('--dataset', '-d', type=str, default='',
                        help='Dataset to use.')
    parser.add_argument('--encoder', '-e', type=str, default='',
                        help='Encoder to use.')
    parser.add_argument('--result_folder', '-o', type=str, default='./output/',
                        help='Result folder')

    parser.add_argument('--imagenet100_path', type=str, default='/data/vision/torralba/datasets/imagenet100', help='Imagenet100 Datasets folder, the directory should contain train/val folders')

    parser.add_argument('--feat_dim', type=int, default=128, help='Encoder feature dimensionality')
    parser.add_argument('--layer_index', type=int, default=5, help='Evaluation layer')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='Learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='When to decay learning rate')

    parser.add_argument('--num_workers', type=int, default=12, help='Number of data loader workers to use')
    parser.add_argument('--log_interval', type=int, default=1, help='Number of iterations between logs')
    parser.add_argument('--gpu', type=int, default='0', help='One GPU to use')

    parser.add_argument('--data_folder', type=str, default='./data', help='Path to data')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes in dataset')

    parser.add_argument('--resize', action='store_true', help='Resize images to 96x96 before augmentation in training')

    opt = parser.parse_args()

    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    opt.gpu = torch.device('cuda', opt.gpu)
    opt.lr_decay_epochs = list(map(int, opt.lr_decay_epochs.split(',')))

    assert opt.dataset != '' or opt.encoder != '', 'Please provide either the encoder or the dataset.'

    if opt.encoder != '':
        assert os.path.isfile(opt.encoder), f'Encoder path {opt.encoder} not found.'
        opt.dataset = None

    else:
        # Make sure dataset contains the name, not the path
        if os.path.isdir(opt.dataset):
            dataset_new = opt.dataset.split('/')[-1]
            print(f'Converting dataset path {opt.dataset} to dataset name {dataset_new}.')
            opt.dataset = dataset_new

        opt.encoder = None

        opt.result_folder = os.path.join('lincls_imagenet/small_scale/', opt.dataset)

    os.makedirs(opt.result_folder, exist_ok=True)

    return opt


def get_data_loaders(opt):
    tfn = []

    if opt.resize:
        tfn += [torchvision.transforms.Resize(96),]

    tfn += [
        torchvision.transforms.RandomResizedCrop(64, scale=(0.08, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ]
    train_transform = torchvision.transforms.Compose(tfn)
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(70),
        torchvision.transforms.CenterCrop(64),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ])

    path_imagenet = opt.imagenet100_path

    assert os.path.isdir(path_imagenet), f'ImageNet-100 path {path_imagenet} does not exist.'

    print(f'Using image data in {path_imagenet}.')
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(path_imagenet, 'train'),
        transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(path_imagenet, 'val'),
        transform=val_transform)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                               num_workers=opt.num_workers, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size,
                                             num_workers=opt.num_workers, pin_memory=True)
    return train_loader, val_loader


def validate(opt, encoder, classifier, val_loader):
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            pred = classifier(encoder(images.to(opt.gpu), layer_index=opt.layer_index).flatten(1)).argmax(dim=1)
            correct += (pred.cpu() == labels).sum().item()
    return correct / len(val_loader.dataset)


def main():
    opt = parse_option()

    torch.cuda.set_device(opt.gpu)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    encoder = SmallAlexNet(feat_dim=opt.feat_dim).to(opt.gpu)
    encoder.eval()
    train_loader, val_loader = get_data_loaders(opt)

    with torch.no_grad():
        sample, _ = train_loader.dataset[0]
        eval_numel = encoder(sample.unsqueeze(0).to(opt.gpu), layer_index=opt.layer_index).numel()
    print(f'Feature dimension: {eval_numel}')

    if opt.encoder is None:
        encoder_checkpoint = os.path.join('encoders/small_scale', opt.dataset, 'encoder.pth')
    else:
        encoder_checkpoint = opt.encoder

    assert os.path.isfile(encoder_checkpoint), f'Encoder checkpoint {encoder_checkpoint} not found.'

    encoder.load_state_dict(torch.load(encoder_checkpoint, map_location=opt.gpu))
    print(f'Loaded checkpoint from {encoder_checkpoint}')

    classifier = nn.Linear(eval_numel, opt.num_classes).to(opt.gpu)

    optim = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=opt.lr_decay_rate,
                                                     milestones=opt.lr_decay_epochs)

    loss_meter = AverageMeter('loss')
    it_time_meter = AverageMeter('iter_time')

    outdir = opt.result_folder

    logfile = os.path.join(outdir, f'log_eval.txt')

    # Initialize python logger
    logging.basicConfig(filename=logfile, level=logging.INFO)


    for epoch in range(opt.epochs):
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()
        for ii, (images, labels) in enumerate(train_loader):
            optim.zero_grad()
            with torch.no_grad():
                feats = encoder(images.to(opt.gpu), layer_index=opt.layer_index).flatten(1)
            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels.to(opt.gpu))
            loss_meter.update(loss, images.shape[0])
            loss.backward()
            optim.step()
            it_time_meter.update(time.time() - t0)
            if ii % opt.log_interval == 0:
                print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(train_loader)}\t{loss_meter}\t{it_time_meter}")
            t0 = time.time()
        scheduler.step()
        val_acc = validate(opt, encoder, classifier, val_loader)
        logging.info(f"Epoch {epoch}/{opt.epochs}\tval_acc {val_acc*100:.4g}%")

    # Save final checkpoint
    fname_out = os.path.join(outdir, 'encoder_classifier.pth')
    torch.save({
        'encoder': encoder.state_dict(),
        'classifier': classifier.state_dict(),
        'layer_index': opt.layer_index,
    },
               fname_out)
    print(f'Wrote output to {fname_out}')

    f = open(os.path.join(outdir, f'val_acc.txt'), "a")
    f.write("Accuracy: {}".format(str(val_acc)))
    f.close()


if __name__ == '__main__':
    main()
