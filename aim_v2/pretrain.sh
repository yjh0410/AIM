# ------------------- Training pipeline -------------------
python -m torch.distributed.run --nproc_per_node=8 --master_port 1904 main_pretrain.py \
        --cuda \
        --distributed \
        --root /scratch014/yangjianhua3/dataset/imagenet_1k/ \
        --dataset imagenet_1k \
        --model aimv2_tiny \
        --batch_size 256 \
        --update_freq 2 \
        --max_epoch 800 \
        --wp_epoch 20 \
        --eval_epoch 20 \
        --base_lr 0.0001 \
        --min_lr 0.0 \
