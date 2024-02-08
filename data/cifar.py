import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10


class CifarDataset(data.Dataset):
    def __init__(self, args, is_train=False, transform=None):
        super().__init__()
        # ----------------- basic parameters -----------------
        self.pixel_mean = [0.5, 0.5, 0.5]
        self.pixel_std =  [0.5, 0.5, 0.5]
        self.is_train  = is_train
        self.num_patches = (args.img_size // args.patch_size) ** 2
        self.image_set = 'train' if is_train else 'val'
        # ----------------- dataset & transforms -----------------
        self.transform = transform if transform is not None else self.build_transform()
        if is_train:
            self.dataset = CIFAR10('cifar_data/', train=True, download=True, transform=self.transform)
        else:
            self.dataset = CIFAR10('cifar_data/', train=False, download=True, transform=self.transform)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, target = self.dataset[index]
        
        # Prefix mask
        prefix_length = random.randint(1, self.num_patches - 1)
        prefix_mask = torch.zeros(self.num_patches).bool()  # False: non-masked; True: masked
        prefix_mask[:prefix_length] = True
            
        return image, target, prefix_mask
    
    def pull_image(self, index):
        # laod data
        image, target = self.dataset[index]

        # denormalize image
        image = image.permute(1, 2, 0).numpy()
        image = (image * self.pixel_std + self.pixel_mean) * 255.
        image = image.astype(np.uint8)
        image = image.copy()

        return image, target

    def build_transform(self):
        if self.is_train:
            transforms = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
        else:
            transforms = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

        return transforms


if __name__ == "__main__":
    import cv2
    import torch
    import argparse
    
    parser = argparse.ArgumentParser(description='Cifar-Dataset')

    # opt
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/VOCdevkit/',
                        help='data root')
    parser.add_argument('--img_size', default=32, type=int,
                        help='input image size.')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='input image size.')
    args = parser.parse_args()

    # dataset
    dataset = CifarDataset(args, is_train=True)  
    print('Dataset size: ', len(dataset))

    for i in range(len(dataset)):
        image, target = dataset.pull_image(i)
        # to BGR
        image = image[..., (2, 1, 0)]

        cv2.imshow('image', image)
        cv2.waitKey(0)
