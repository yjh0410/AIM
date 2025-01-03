import torch.utils.data as data
import torch

from .cifar import CifarDataset
from .imagenet import ImageNet1KDataset


def build_dataset(args, transform=None, is_train=False):
    # ----------------- CIFAR dataset -----------------
    if args.dataset == 'cifar10':
        args.num_classes = 10
        return CifarDataset(args, is_train, transform)
    
    # ----------------- ImageNet dataset -----------------
    elif args.dataset == 'imagenet_1k':
        args.num_classes = 1000
        return ImageNet1KDataset(args, is_train, transform)
        

def build_dataloader(args, dataset, is_train=False):
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train, drop_last=is_train)
    else:
        sampler = None

    per_gpu_batch = args.batch_size // args.world_size
    if is_train:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=per_gpu_batch, shuffle=(sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=sampler)
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=per_gpu_batch, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, sampler=sampler)

    return dataloader
