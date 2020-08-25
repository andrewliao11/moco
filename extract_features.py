"""
sgpu python extract_features.py -a resnet50 -s train --pretrained moco_v2_800ep_pretrain.pth.tar /scratch/gobi1/datasets/imagenet
"""
import os
import time
import argparse
import random
import torch
import warnings
import numpy as np
from tqdm import tqdm

import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from my_dataset import ImageFolder
from main_lincls import AverageMeter, ProgressMeter
import ipdb


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
    

parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-s', '--split', type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    model.fc = torch.nn.Identity()

    print("=> loading checkpoint '{}'".format(args.pretrained))
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    state_dict = checkpoint['state_dict']
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
        

    #model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(state_dict)
    model.cuda()

    valdir = os.path.join(args.data, args.split)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), return_path=True),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate(val_loader, model, args)


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix='Test: ')


    # switch to evaluate mode
    model.eval()

    path_arr = []
    output_arr = []
    target_arr = []
    with torch.no_grad():
        end = time.time()
        for i, (paths, images, targets) in tqdm(enumerate(val_loader)):
            images = images.cuda()

            # compute output
            output = model(images)
            path_arr += [j for j in paths]
            output_arr += [j for j in output.cpu().numpy()]
            target_arr += [j for j in targets.cpu().numpy()]
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)


    path_arr = np.array(path_arr)
    output_arr = np.array(output_arr)
    target_arr = np.array(target_arr)

    idx = list(val_loader.dataset.idx_to_class.keys())
    idx.sort()
    idx_to_class = np.array([val_loader.dataset.idx_to_class[i] for i in idx])

    np.savez("/scratch/gobi2/andrewliao/moco/moco-v2-{}".format(args.split),
             path=path_arr,
             output=output_arr,
             target=target_arr,
             idx_to_class=idx_to_class)


if __name__ == '__main__':
    main()
