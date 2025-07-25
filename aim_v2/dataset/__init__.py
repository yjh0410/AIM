import torch
import torch.utils.data as data

from .cifar import CifarDataset
from .imagenet import ImageNet1KDataset


def build_dataset(
        args,
        img_size: int = 256,
        patch_size: int = 16,
        max_length: int = 512,
        transform = None,
        is_train = False,
        ):
    # ----------------- CIFAR dataset -----------------
    if args.dataset == 'cifar10':
        args.num_classes = 10
        return CifarDataset(
            img_size = img_size,
            patch_size = patch_size,
            is_train = is_train,
            max_length = max_length,
            transform = transform,
            )
    
    # ----------------- ImageNet dataset -----------------
    elif args.dataset == 'imagenet_1k' or args.dataset == 'in1k':
        args.num_classes = 1000
        return ImageNet1KDataset(
            args = args,
            img_size = img_size,
            patch_size = patch_size,
            is_train = is_train,
            max_length = max_length,
            transform = transform,
            )
        
    else:
        print("Unknown dataset: {}".format(args.dataset))
    

def build_dataloader(args, dataset, is_train=False, collate_fn=None):
    if is_train:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=args.world_size, rank=args.global_rank, shuffle=True, seed=args.seed,
            )

        dataloader = torch.utils.data.DataLoader(
            dataset, sampler = sampler,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            pin_memory  = True,
            drop_last   = True,
            collate_fn  = collate_fn,
        )

    else:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=args.world_size, rank=args.global_rank, shuffle=False,
            )
            
        dataloader = torch.utils.data.DataLoader(
            dataset, sampler = sampler,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            pin_memory  = False,
            drop_last   = False,
            collate_fn  = collate_fn,
        )


    return dataloader
