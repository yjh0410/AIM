# ------------------- Model setting -------------------
MODEL=$1
BATCH_SIZE=$2
DATASET=$3
DATASET_ROOT=$4
PRETRAINED_MODEL=$5
WORLD_SIZE=$6
MASTER_PORT=$7
RESUME=$8

# ------------------- Training setting -------------------
OPTIMIZER="adamw"
WEIGHT_DECAY=0.05

if [ $MODEL == "vit_h" ]; then
    MAX_EPOCH=50
    WP_EPOCH=5
    EVAL_EPOCH=5
    BASE_LR=0.001
    MIN_LR=0.0
    LAYER_DECAY=0.75
    DROP_PATH=0.3
    MIXUP=0.8
    CUTMIX=1.0

elif [ $MODEL == "vit_l" ]; then
    MAX_EPOCH=50
    WP_EPOCH=5
    EVAL_EPOCH=5
    BASE_LR=0.001
    MIN_LR=0.0
    LAYER_DECAY=0.75
    DROP_PATH=0.2
    MIXUP=0.8
    CUTMIX=1.0

elif [ $MODEL == "vit_s" ]; then
    MAX_EPOCH=100
    WP_EPOCH=5
    EVAL_EPOCH=5
    BASE_LR=0.001
    MIN_LR=0.0
    LAYER_DECAY=0.75
    DROP_PATH=0.1
    MIXUP=0.8
    CUTMIX=1.0

elif [ $MODEL == "vit_t" ]; then
    MAX_EPOCH=100
    WP_EPOCH=5
    EVAL_EPOCH=5
    BASE_LR=0.001
    MIN_LR=0.0
    LAYER_DECAY=1.0
    DROP_PATH=0.1
    MIXUP=0.8
    CUTMIX=1.0

else
    MAX_EPOCH=100
    WP_EPOCH=5
    EVAL_EPOCH=5
    BASE_LR=0.0005
    MIN_LR=0.0
    LAYER_DECAY=0.65
    DROP_PATH=0.1
    MIXUP=0.8
    CUTMIX=0.0

fi

# ------------------- Dataset config -------------------
if [[ $DATASET == "cifar10" || $DATASET == "cifar100" ]]; then
    # Data root
    ROOT="none"
    # Image config
    IMG_SIZE=32
    PATCH_SIZE=2
elif [[ $DATASET == "imagenet_1k" || $DATASET == "imagenet_22k" ]]; then
    # Data root
    ROOT="path/to/imagenet"
    # Image config
    IMG_SIZE=224
    PATCH_SIZE=16
elif [[ $DATASET == "custom" ]]; then
    # Data root
    ROOT="path/to/custom"
    # Image config
    IMG_SIZE=224
    PATCH_SIZE=16
else
    echo "Unknown dataset!!"
    exit 1
fi


# ------------------- Training pipeline -------------------
if (( $WORLD_SIZE >= 1 && $WORLD_SIZE <= 8 )); then
    python -m torch.distributed.run --nproc_per_node=${WORLD_SIZE} --master_port ${MASTER_PORT} main_finetune.py \
            --cuda \
            --distributed \
            --root ${DATASET_ROOT} \
            --dataset ${DATASET} \
            --model ${MODEL} \
            --batch_size ${BATCH_SIZE} \
            --img_size ${IMG_SIZE} \
            --patch_size ${PATCH_SIZE} \
            --drop_path ${DROP_PATH} \
            --max_epoch ${MAX_EPOCH} \
            --wp_epoch ${WP_EPOCH} \
            --eval_epoch ${EVAL_EPOCH} \
            --optimizer ${OPTIMIZER} \
            --base_lr ${BASE_LR} \
            --min_lr ${MIN_LR} \
            --layer_decay ${LAYER_DECAY} \
            --weight_decay ${WEIGHT_DECAY} \
            --reprob 0.0 \
            --mixup ${MIXUP} \
            --cutmix ${CUTMIX} \
            --resume ${RESUME} \
            --pretrained ${PRETRAINED_MODEL}
else
    echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
          multi-card training mode, which is currently unsupported."
    exit 1
fi