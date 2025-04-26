import os
import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

import sys
abso_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abso_path)
from tokenizer import load_hf_tokenizer


class CifarDataset(data.Dataset):
    def __init__(self, args, is_train=False, transform=None):
        super().__init__()
        # ----------------- basic parameters -----------------
        self.pixel_mean = [0.0, 0.0, 0.0]
        self.pixel_std =  [1.0, 1.0, 1.0]
        self.is_train  = is_train
        self.num_patches = (args.img_size // args.patch_size) ** 2
        self.cifar10_classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ]

        # ----------------- dataset & transforms -----------------
        self.image_set = 'train' if is_train else 'val'
        self.transform = transform if transform is not None else self.build_transform()
        abso_path = os.path.dirname(os.path.abspath(__file__))
        if is_train:
            self.dataset = CIFAR10(os.path.join(abso_path, 'cifar_data/'), train=True, download=True, transform=self.transform)
        else:
            self.dataset = CIFAR10(os.path.join(abso_path, 'cifar_data/'), train=False, download=True, transform=self.transform)

        llama3_tokenizer_config_path = os.path.join(abso_path, 'llama3_tokenizer_config')
        self.tokenizer = load_hf_tokenizer(checkpoint=llama3_tokenizer_config_path)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, cls_idx = self.dataset[index]

        # Prefix mask
        prefix_length = random.randint(1, self.num_patches - 1)
        prefix_mask = torch.zeros(self.num_patches).bool()  # True: non-masked; False: masked
        prefix_mask[:prefix_length] = True

        # create a text
        text = "A photo of the {}.".format(self.cifar10_classes[cls_idx])
        token_ids = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
        token_mask = np.ones_like(token_ids, dtype=np.int8).tolist()

        pad_token_id = self.tokenizer.pad_token_id

        return image, token_ids, token_mask, prefix_mask, cls_idx, pad_token_id
    
    def pull_image(self, index):
        # laod data
        image, cls_idx = self.dataset[index]

        # convert image tensor into image numpy
        image = image.permute(1, 2, 0).numpy()
        image = (image * self.pixel_std + self.pixel_mean) * 255.
        image = np.clip(image, 0., 255.).astype(np.uint8)
        image = image.copy()

        return image, cls_idx

    def build_transform(self):
        if self.is_train:
            transforms = T.Compose([T.ToTensor(), T.Normalize(self.pixel_mean, self.pixel_std)])
        else:
            transforms = T.Compose([T.ToTensor(), T.Normalize(self.pixel_mean, self.pixel_std)])

        return transforms

if __name__ == "__main__":
    import cv2
    import argparse
    
    parser = argparse.ArgumentParser(description='Cifar10 Dataset')

    # opt
    parser.add_argument('--img_size', default=32, type=int,
                        help='data root')
    parser.add_argument('--patch_size', default=8, type=int,
                        help='data root')
    parser.add_argument('--is_train', action="store_true", default=False,
                        help='mixup augmentation.')

    args = parser.parse_args()

    # dataset
    dataset = CifarDataset(args, is_train=args.is_train)  
    print('Dataset size: ', len(dataset))

    for i in range(len(dataset)):
        image, token_ids, token_mask, prefix_mask, cls_idx, pad_token_id = dataset[i]
        text = dataset.tokenizer.decode(token_ids)

        print(" - token ids: ", token_ids)
        print(" - decoded texts ids: ", text)

        # convert image tensor into image numpy
        image = image.permute(1, 2, 0).numpy()
        image = (image * dataset.pixel_std + dataset.pixel_mean) * 255.
        image = np.clip(image, 0., 255.).astype(np.uint8)
        image = image.copy()

        # to BGR
        image = image[..., (2, 1, 0)]

        cv2.imshow(text, image)
        cv2.waitKey(0)
        cv2.destroyWindow(text)
