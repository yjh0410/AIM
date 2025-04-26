# autoregressive_image_modeling

Try to reproduce Visual Autoregressive Image Modeling proposed by Apple


# Pretrain
```Shell
python main_pretrain.py \
    --cuda \
    --dataset cifar10 \
    --model vit_t \
    --img_size 32 \
    --patch_size 4 \
    --batch_size 256 \
    --max_epoch 400 \
    --wp_epoch 40 \
    --eval_epoch 20 \
    --base_lr 0.00015 \
    --min_lr 0 \

```
# Finetune 
```Shell
python main_finetune.py \
    --cuda \
    --dataset cifar10 \
    --model vit_t \
    --img_size 32 \
    --patch_size 4 \
    --batch_size 256 \
    --max_epoch 100 \
    --wp_epoch 5 \
    --eval_epoch 10 \
    --base_lr 0.001 \
    --min_lr 0 \
    --layer_decay 1.0 \
    --drop_path 0.1 \
    --mixup 0.8 \
    --cutmix 0.0 \
    --pretrained path/to/pretrained

```