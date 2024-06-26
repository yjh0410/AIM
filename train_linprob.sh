# ------------------- Args setting -------------------
MODEL=$1
MODEL_T=$2
BATCH_SIZE=$3
DATASET=$4
DATASET_ROOT=$5
WORLD_SIZE=$6
RESUME=$7

# ------------------- Training setting -------------------
## Epoch
MAX_EPOCH=100
WP_EPOCH=5
EVAL_EPOCH=5

## Optimizer
BASE_LR=0.1
MIN_LR=0.0
WEIGHT_DECAY=0.05

# ------------------- Dataset setting -------------------
DATASET="cifar10"
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
    python -m torch.distributed.run --nproc_per_node=${WORLD_SIZE} --master_port 1668 main_linprobe.py \
            --cuda \
            --distributed \
            --root ${ROOT} \
            --dataset ${DATASET} \
            -m ${MODEL} \
            --batch_size ${BATCH_SIZE} \
            --img_size ${IMG_SIZE} \
            --patch_size ${PATCH_SIZE} \
            --max_epoch ${MAX_EPOCH} \
            --wp_epoch ${WP_EPOCH} \
            --eval_epoch ${EVAL_EPOCH} \
            --base_lr ${BASE_LR} \
            --min_lr ${MIN_LR} \
            --weight_decay ${WEIGHT_DECAY} \
            --pretrained ${PRETRAINED_MODEL}
else
    echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
          multi-card training mode, which is currently unsupported."
    exit 1
fi