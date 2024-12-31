# AIM: Autoregressive Image Models
**Unofficial PyTorch implementation of Autoregressive Image Models**

Due to limited resources, I only test my randomly designed `ViT-Tiny` on the `CIFAR10` dataset. It is not my goal to reproduce AIM perfectly, but my implementation is aligned with the official AIM as much as possible so that users can learn `AIM` quickly and accurately.

## 1. Pretrain
We have kindly provided the bash script `train_pretrain.sh` file for pretraining. You can modify some hyperparameters in the script file according to your own needs.

```Shell
bash train_pretrain.sh
```

## 2. Finetune
We have kindly provided the bash script `train_finetune.sh` file for finetuning. You can modify some hyperparameters in the script file according to your own needs.

```Shell
bash train_finetune.sh
```

## 3. Evaluate 
- Evaluate the `top1 & top5` accuracy of `ViT-Tiny` on CIFAR10 dataset:
```Shell
python train_finetune.py --dataset cifar10 -m vit_tiny --batch_size 256 --img_size 32 --patch_size 2 --eval --resume path/to/vit_tiny_cifar10.pth
```

- Evaluate the `top1 & top5` accuracy of `ViT-Tiny` on ImageNet-1K dataset:
```Shell
python train_finetune.py --dataset imagenet_1k -m vit_tiny --batch_size 256 --img_size 224 --patch_size 16 --eval --resume path/to/vit_tiny_imagenet_1k.pth
```


## 4. Visualize Image Reconstruction
This function is not complete yet.
<!-- - Evaluate `MAE-ViT-Tiny` on CIFAR10 dataset:
```Shell
python train_pretrain.py --dataset cifar10 -m mae_vit_tiny --resume path/to/mae_vit_tiny_cifar10.pth --img_size 32 --patch_size 2 --eval --batch_size 1
```

- Evaluate `MAE-ViT-Tiny` on ImageNet-1K dataset:
```Shell
python train_pretrain.py --dataset imagenet_1k -m mae_vit_tiny --resume path/to/mae_vit_tiny_imagenet_1k.pth --img_size 224 --patch_size 16 --eval --batch_size 1
``` -->


## 5. Experiments
### 5.1 AIM pretrain
- Visualization on CIFAR10 validation

This experiment is not complete yet.

<!-- Masked Image | Original Image | Reconstructed Image

![image](./img_files/visualize_cifar10_mae_vit_nano.png)

- Visualization on ImageNet validation

... -->


### 5.2 Finutune
- On CIFAR10

|  Model   |  AIM pretrained  | Epoch | Top 1     | Weight |  MAE weight  |
|  :---:   |       :---:      | :---: | :---:     | :---:  |    :---:     |
| ViT-Tiny |        Yes       | 100   |           |        |              |


## 6. Acknowledgment
Thanks to the official implementation of [AIM](https://github.com/apple/ml-aim).
