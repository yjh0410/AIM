from copy import deepcopy
import os
import time
import math
import argparse
import datetime

# ---------------- Timm compoments ----------------
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# ---------------- Torch compoments ----------------
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------- Dataset compoments ----------------
from dataset import build_dataset, build_dataloader

# ---------------- Model compoments ----------------
from models import build_model
from models.config import build_config

# ---------------- Utils compoments ----------------
from utils import lr_decay
from utils import distributed_utils
from utils.misc import setup_seed, print_rank_0, load_model, save_model, CollateFunc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.lr_scheduler import LinearWarmUpLrScheduler
from utils.com_flops_params import FLOPs_and_Params

# ---------------- Training engine ----------------
from engine_finetune import train_one_epoch, evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic settings
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate model.')
        
    # Epoch settings
    parser.add_argument('--wp_epoch', type=int, default=5, 
                        help='warmup epoch for finetune with MAE pretrained')
    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='start epoch for finetune with MAE pretrained')
    parser.add_argument('--max_epoch', type=int, default=50, 
                        help='max epoch')
    parser.add_argument('--eval_epoch', type=int, default=5, 
                        help='max epoch')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    parser.add_argument('--num_classes', type=int, default=None, 
                        help='number of classes.')
    
    # Model settings
    parser.add_argument('--model', type=str, default='vit_tiny',
                        help='model name')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='load pretrained weight.')
    parser.add_argument('--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema.')
    parser.add_argument('--drop_path', type=float, default=0.1,
                        help='drop_path')
    
    # Optimizer settings
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size on all GPUs')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='learning rate for training model')
    parser.add_argument('--min_lr', type=float, default=0.0,
                        help='the final lr')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--update_freq', type=int, default=2,
                        help='batch size per gpu')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    
    # Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    # DDP
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='the number of local rank.')

    return parser.parse_args()

    
def main():
    args = parse_args()
    # set random seed
    setup_seed(args.seed)

    
    # ------------------------- Build DDP environment -------------------------
    ## LOCAL_RANK is the global GPU number tag, the value range is [0, world_size - 1].
    ## LOCAL_PROCESS_RANK is the number of the GPU of each machine, not global.
    local_rank = local_process_rank = -1
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))
        try:
            # Multiple Mechine & Multiple GPUs (world size > 8)
            local_rank = torch.distributed.get_rank()
            local_process_rank = int(os.getenv('LOCAL_PROCESS_RANK', '0'))
        except:
            # Single Mechine & Multiple GPUs (world size <= 8)
            local_rank = local_process_rank = torch.distributed.get_rank()
    args.world_size = distributed_utils.get_world_size()
    args.global_rank = distributed_utils.get_rank()
    print('World size: {}'.format(distributed_utils.get_world_size()))
    print_rank_0(args, local_rank)
    print("LOCAL RANK: ", local_rank)
    print("LOCAL_PROCESS_RANL: ", local_process_rank)


    # ------------------------- Build CUDA -------------------------
    if torch.cuda.is_available():
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        print('There is no available GPU.')
        args.cuda = False
        device = torch.device("cpu")


    # ------------------------- Build Tensorboard -------------------------
    tblogger = None
    if local_rank <= 0 and args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        time_stamp = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, time_stamp)
        os.makedirs(log_path, exist_ok=True)
        tblogger = SummaryWriter(log_path)


    # ------------------------- Build Model config -------------------------
    model_cfg = build_config(args.model)
    model_cfg.vit_is_causal = False


    # ------------------------- Build Dataset -------------------------
    if 'cifar' in args.dataset:
        model_cfg.vit_img_size   = 32
        model_cfg.vit_patch_size = 8

    train_dataset = build_dataset(
        args = args,
        img_size = model_cfg.vit_img_size,
        patch_size = model_cfg.vit_patch_size,
        max_length = model_cfg.lm_max_length,
        is_train = True,
        )
    valid_dataset = build_dataset(
        args,
        img_size = model_cfg.vit_img_size,
        patch_size = model_cfg.vit_patch_size,
        max_length = model_cfg.lm_max_length,
        is_train = False,
        )
    print_rank_0('\n =================== Dataset Information ===================', local_rank)
    print_rank_0(' - Train dataset size : {}'.format(len(train_dataset)), local_rank)
    print_rank_0(' - Valid dataset size : {}'.format(len(valid_dataset)), local_rank)


    # ------------------------- Build Dataloader -------------------------
    train_dataloader = build_dataloader(args, train_dataset, is_train=True, collate_fn=CollateFunc())
    valid_dataloader = build_dataloader(args, valid_dataset, is_train=False, collate_fn=CollateFunc())
    print_rank_0('\n =================== Epoch Information ===================', local_rank)
    print_rank_0(' - Epoch size : {}'.format(len(train_dataloader)), local_rank)
    print_rank_0(' - Train epochs : {}'.format(args.max_epoch), local_rank)
    print_rank_0(' - Train iterations : {}'.format(args.max_epoch * len(train_dataloader)), local_rank)
    print_rank_0(' - Warmup epochs : {}'.format(args.wp_epoch), local_rank)
    print_rank_0(' - Warmup iterations : {}'.format(args.wp_epoch * len(train_dataloader)), local_rank)


    # ------------------------- Mixup augmentation config -------------------------
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print_rank_0(" => Mixup is activated!", local_rank)
        mixup_fn = Mixup(mixup_alpha     = args.mixup,
                         cutmix_alpha    = args.cutmix,
                         cutmix_minmax   = args.cutmix_minmax,
                         prob            = args.mixup_prob,
                         switch_prob     = args.mixup_switch_prob,
                         mode            = args.mixup_mode,
                         label_smoothing = args.smoothing,
                         num_classes     = args.num_classes)


    # ------------------------- Build Model -------------------------
    model = build_model(args, model_cfg, model_type='cls')
    model.train().to(device)
    print(model)
    if local_rank <= 0:
        FLOPs_and_Params(model=deepcopy(model).eval(), size=model_cfg.vit_img_size)
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()


    # ------------------------- Build DDP Model -------------------------
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    # ------------------------- Build Optimzier -------------------------
    param_groups = lr_decay.param_groups_lrd(model_without_ddp, 0.05, [], args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.base_lr)
    loss_scaler = NativeScaler()
    print(' - Base lr: ', args.base_lr)
    print(' - Mun  lr: ', args.min_lr)


    # ------------------------- Build Lr Scheduler -------------------------
    lr_scheduler_warmup = LinearWarmUpLrScheduler(args.base_lr, wp_iter=args.wp_epoch * len(train_dataloader))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max = (args.max_epoch - args.wp_epoch - 1) * len(train_dataloader), eta_min = args.min_lr)


    # ------------------------- Build Criterion -------------------------
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    load_model(args=args, model_without_ddp=model_without_ddp,
               optimizer=optimizer, lr_scheduler=lr_scheduler, loss_scaler=loss_scaler)

    # ------------------------- Eval before Train Pipeline -------------------------
    if args.eval:
        print('evaluating ...')
        test_stats = evaluate(valid_dataloader, model, device, local_rank)
        print('Eval Results: [loss: %.2f][acc1: %.2f][acc5 : %.2f]' %
                (test_stats['loss'], test_stats['acc1'], test_stats['acc5']), flush=True)
        return

    # Path to save model
    output_dir = os.path.join("weights/", args.dataset, "finetune", args.model + '_patch{}'.format(model_cfg.vit_patch_size))
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    # ------------------------- Training Pipeline -------------------------
    start_time = time.time()
    max_accuracy = -1.0
    print_rank_0("=============== Start training for {} epochs ===============".format(args.max_epoch), local_rank)
    for epoch in range(args.start_epoch, args.max_epoch):
        if args.distributed:
            train_dataloader.batch_sampler.sampler.set_epoch(epoch)

        # train one epoch
        train_one_epoch(args = args,
                        device = device,
                        model = model,
                        data_loader = train_dataloader,
                        epoch = epoch,
                        optimizer = optimizer,
                        lr_scheduler = lr_scheduler,
                        lr_scheduler_warmup = lr_scheduler_warmup,
                        loss_scaler = loss_scaler,
                        local_rank = local_rank,
                        criterion = criterion,
                        tblogger = tblogger,
                        mixup_fn = mixup_fn,
                        )

        # Evaluate
        if (epoch % args.eval_epoch) == 0 or (epoch + 1 == args.max_epoch):
            test_stats = evaluate(valid_dataloader, model, device, local_rank)
            print_rank_0(f"Accuracy of the network on the {len(valid_dataset)} test images: {test_stats['acc1']:.1f}%", local_rank)
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print_rank_0(f'Max accuracy: {max_accuracy:.2f}%', local_rank)

            # Save model
            if local_rank <= 0:
                print('- saving the model after {} epochs ...'.format(epoch))
                save_model(args = args,
                           epoch = epoch,
                           model_without_ddp = model_without_ddp,
                           optimizer = optimizer,
                           lr_scheduler = lr_scheduler,
                           loss_scaler = loss_scaler,
                           metric = test_stats["acc1"],
                           best_metric = max_accuracy,
                           )
        if args.distributed:
            dist.barrier()

        if tblogger is not None:
            tblogger.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            tblogger.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            tblogger.add_scalar('perf/test_loss', test_stats['loss'], epoch)
        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()