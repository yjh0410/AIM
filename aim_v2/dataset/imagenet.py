import os
import PIL
import random
import numpy as np
import json

import torch
from timm.data import create_transform
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import sys
abso_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abso_path)
from tokenizer import load_hf_tokenizer


class ImageNet1KDataset(data.Dataset):
    def __init__(self,
                 args,
                 img_size: int = 256,
                 patch_size: int = 16,
                 is_train = False,
                 max_length = 512,
                 transform = None,
                 ):
        super().__init__()
        super().__init__()
        # ----------------- basic parameters -----------------
        self.is_train  = is_train
        self.num_patches = (img_size // patch_size) ** 2
        self.max_length = max_length

        assert max_length > self.num_patches

        # load imagenet class names
        abso_path = os.path.dirname(os.path.abspath(__file__))
        json_file = os.path.join(abso_path, 'imagenet_1k_classes.json')
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.classes = data

        # ----------------- dataset & transforms -----------------
        self.image_set = 'train' if is_train else 'val'
        self.data_path = os.path.join(args.root, self.image_set)
        self.transform = transform if transform is not None else self.build_transform(args)
        self.dataset = ImageFolder(root=self.data_path, transform=self.transform)

        llama3_tokenizer_config_path = os.path.join(abso_path, 'llama3_tokenizer_config')
        self.tokenizer = load_hf_tokenizer(checkpoint=llama3_tokenizer_config_path)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, cls_idx = self.dataset[index]

        # Prefix mask
        prefix_length = random.randint(1, self.num_patches - 1)
        prefix_mask = torch.zeros(self.num_patches).bool()  # False: non-masked; True: masked
        prefix_mask[:prefix_length] = True

        # create a text
        text = "A photo of the {}.".format(self.classes[cls_idx])
        input_ids = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]

        assert self.max_length >= self.num_patches + len(input_ids)

        # prepare attention mask
        attention_mask = np.zeros(self.max_length, dtype=np.float32)
        attention_mask[:self.num_patches + len(input_ids)] = 1.0
        attention_mask = attention_mask.tolist()

        # pad the input token ids
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - self.num_patches - len(input_ids))

        return image, input_ids, attention_mask, prefix_mask, cls_idx
    
    def pull_image(self, index):
        # laod data
        image, cls_idx = self.dataset[index]

        # denormalize image
        image = image.permute(1, 2, 0).numpy()
        image = image * 255.
        image = np.clip(image, 0., 255.).astype(np.uint8)
        image = image.copy()

        return image, cls_idx

    def build_transform(self, args):
        if self.is_train:
            transforms = create_transform(input_size    = args.img_size,
                                          is_training   = True,
                                          color_jitter  = args.color_jitter,
                                          auto_augment  = args.aa,
                                          interpolation = 'bicubic',
                                          re_prob       = args.reprob,
                                          re_mode       = args.remode,
                                          re_count      = args.recount,
                                          mean          = [0.0, 0.0, 0.0],
                                          std           = [1.0, 1.0, 1.0],
                                          )
        else:
            t = []
            if args.img_size <= 224:
                crop_pct = 224 / 256
            else:
                crop_pct = 1.0
            size = int(args.img_size / crop_pct)
            t.append(
                T.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(T.CenterCrop(args.img_size))
            t.append(T.ToTensor())
            transforms = T.Compose(t)

        return transforms


if __name__ == "__main__":
    import cv2
    import torch
    import argparse
    
    parser = argparse.ArgumentParser(description='ImageNet-Dataset')

    # opt
    parser.add_argument('--root', default='D:/dataset/imagenet_1k/',
                        help='data root')
    parser.add_argument('--img_size', default=256, type=int,
                        help='input image size.')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='input image size.')
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--is_train', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    args = parser.parse_args()
  
    # Dataset
    dataset = ImageNet1KDataset(
        args=args,
        img_size = 256,
        patch_size = 16,
        is_train = False,
        max_length = 288,
        )  
    print('Dataset size: ', len(dataset))

    for i in range(len(dataset)):
        image, input_ids, attention_mask, prefix_mask, cls_idx = dataset[i]
        text = dataset.tokenizer.decode(input_ids)

        print(" - token ids: ", input_ids)
        print(" - decoded texts ids: ", text)

        # convert image tensor into image numpy
        image = image.permute(1, 2, 0).numpy()
        image = image * 255.
        image = np.clip(image, 0., 255.).astype(np.uint8)
        image = image.copy()

        # to BGR
        image = image[..., (2, 1, 0)]

        cv2.imshow(text, image)
        cv2.waitKey(0)
        cv2.destroyWindow(text)
