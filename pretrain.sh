# ------------------- Args setting -------------------
MODEL=$1
BATCH_SIZE=$2
DATASET=$3
DATASET_ROOT=$4
WORLD_SIZE=$5
RESUME=$6

# ------------------- Training setting -------------------
# Optimizer config
OPTIMIZER="adamw"
BASE_LR=0.00015
MIN_LR=0
WEIGHT_DECAY=0.05
# Epoch
MAX_EPOCH=800
WP_EPOCH=40
EVAL_EPOCH=20

# ------------------- Dataset setting -------------------
if [[ $DATASET == "cifar10" ]]; then
    IMG_SIZE=32
    PATCH_SIZE=2
    NUM_CLASSES=10
elif [[ $DATASET == "imagenet_1k" ]]; then
    IMG_SIZE=224
    PATCH_SIZE=16
    NUM_CLASSES=1000
else
    echo "Unknown dataset!!"
    exit 1
fi

# ------------------- Training pipeline -------------------
if (( $WORLD_SIZE >= 1 && $WORLD_SIZE <= 8 )); then
    python -m torch.distributed.run --nproc_per_node=${WORLD_SIZE} --master_port 1700 main_pretrain.py \
            --cuda \
            --distributed \
            --root ${DATASET_ROOT} \
            --dataset ${DATASET} \
            --model ${MODEL} \
            --resume ${RESUME} \
            --batch_size ${BATCH_SIZE} \
            --img_size ${IMG_SIZE} \
            --patch_size ${PATCH_SIZE} \
            --max_epoch ${MAX_EPOCH} \
            --wp_epoch ${WP_EPOCH} \
            --eval_epoch ${EVAL_EPOCH} \
            --optimizer ${OPTIMIZER} \
            --base_lr ${BASE_LR} \
            --min_lr ${MIN_LR} \
            --weight_decay ${WEIGHT_DECAY}
else
    echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
          multi-card training mode, which is currently unsupported."
    exit 1
fi