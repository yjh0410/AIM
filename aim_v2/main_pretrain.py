import os
import time
import datetime
import argparse

# ---------------- Torch compoments ----------------
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------- Torchvision compoments ----------------
import torchvision.transforms as transforms

# ---------------- Dataset compoments ----------------
from dataset import build_dataset, build_dataloader
from models import build_model

# ---------------- Utils compoments ----------------
from utils import distributed_utils
from utils.misc import setup_seed
from utils.misc import load_model, save_model
from utils.misc import print_rank_0, CollateFunc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.optimizer import build_optimizer
from utils.lr_scheduler import LinearWarmUpLrScheduler

# ---------------- Training engine ----------------
from engine_pretrain import train_one_epoch, evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    # Input
    parser.add_argument('--img_size', type=int, default=224,
                        help='input image size.')    
    parser.add_argument('--img_dim', type=int, default=3,
                        help='3 for RGB; 1 for Gray.')    
    parser.add_argument('--patch_size', type=int, default=16,
                        help='patch_size.')    
    # Basic
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size on all GPUs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--output_dir', type=str, default='weights/',
                        help='path to save trained model.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    # Epoch
    parser.add_argument('--wp_epoch', type=int, default=100, 
                        help='warmup epoch for finetune with MAE pretrained')
    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='start epoch for finetune with MAE pretrained')
    parser.add_argument('--eval_epoch', type=int, default=20, 
                        help='warmup epoch for finetune with MAE pretrained')
    parser.add_argument('--max_epoch', type=int, default=4000, 
                        help='max epoch')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    parser.add_argument('--num_classes', type=int, default=None, 
                        help='number of classes.')
    # Model
    parser.add_argument('--model', type=str, default='aimv2_t',
                        help='model name of AIMv2')
    parser.add_argument('--resume', default=None, type=str,
                        help='keep training')
    # Optimizer
    parser.add_argument('-opt', '--optimizer', type=str, default='adamw',
                        help='sgd, adam')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--base_lr', type=float, default=0.001,
                        help='learning rate for training model')
    parser.add_argument('--min_lr', type=float, default=0,
                        help='the final lr')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Clip gradient norm (default: None, no clipping)')
    # Loss
    parser.add_argument('--norm_pix_loss', action='store_true', default=False,
                        help='normalize pixels before computing loss.')
    # DDP
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='the number of local rank.')

    return parser.parse_args()

    
def main():
    args = parse_args()
    # set random seed
    setup_seed(args.seed)

    # Path to save model
    if args.output_dir is not None:
        args.output_dir = args.model + "_" + args.output_dir
    else:
        args.output_dir = args.model
        
    output_dir = os.path.join("weights/", args.dataset, "pretrain", args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    
    
    # ------------------------- Build DDP environment -------------------------
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


    # ------------------------- Build CUDA -------------------------
    if args.cuda:
        if torch.cuda.is_available():
            cudnn.benchmark = True
            device = torch.device("cuda")
        else:
            print('There is no available GPU.')
            args.cuda = False
            device = torch.device("cpu")
    else:
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


    # ------------------------- Build Transforms -------------------------
    train_transform = None
    if 'cifar' not in args.dataset:
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.img_size, scale=(0.4, 1.0), ratio=(0.75, 1.33), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    

    # ------------------------- Build Dataset -------------------------
    train_dataset = build_dataset(args, transform=train_transform, is_train=True)
    valid_dataset = build_dataset(args, transform=None, is_train=False)
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


   # ------------------------- Build Model -------------------------
    print_rank_0('\n =================== Model Information ===================', local_rank)
    model = build_model(args, model_type='aim')
    model.train().to(device)


    # ------------------------- Build DDP Model -------------------------
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    # ------------------------- Build Optimzier -------------------------
    print_rank_0('\n =================== Optimizer Information ===================', local_rank)
    optimizer = build_optimizer(model_without_ddp, base_lr=args.base_lr, weight_decay=args.weight_decay)
    print_rank_0(' - Base lr: {}'.format(args.base_lr), local_rank)
    print_rank_0(' - Mun  lr: {}'.format(args.min_lr), local_rank)


    # ------------------------- Build Lr Scheduler -------------------------
    print_rank_0('\n =================== Lr Scheduler Information ===================', local_rank)
    lr_scheduler_warmup = LinearWarmUpLrScheduler(args.base_lr, wp_iter=args.wp_epoch * len(train_dataloader))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max = (args.max_epoch - args.wp_epoch - 1) * len(train_dataloader), eta_min = args.min_lr)
    print_rank_0(' - T_max: {}'.format((args.max_epoch - args.wp_epoch - 1) * len(train_dataloader)), local_rank)
    print_rank_0(' - eta_min: {}'.format(args.min_lr), local_rank)


    # ------------------------- Build Loss scaler -------------------------
    loss_scaler = NativeScaler()
    load_model(args=args, model_without_ddp=model_without_ddp,
               optimizer=optimizer, lr_scheduler=lr_scheduler, loss_scaler=loss_scaler)


    # ------------------------- Training Pipeline -------------------------
    start_time = time.time()
    min_loss = float("inf")
    print_rank_0("\n =================== Start training for {} epochs ===================".format(args.max_epoch), local_rank)
    for epoch in range(args.start_epoch, args.max_epoch):
        if args.distributed:
            train_dataloader.batch_sampler.sampler.set_epoch(epoch)
        
        # Train one epoch
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
                        tblogger = tblogger,
                        )

        # Evaluate
        if epoch % args.eval_epoch == 0 or epoch + 1 == args.max_epoch:
            print_rank_0("\n =================== Evaluation pipeline ===================", local_rank)
            test_stats = evaluate(valid_dataloader, model, device, local_rank)
            print_rank_0(f" - Loss of the network on the {len(valid_dataset)} test images: {test_stats['loss']:.1f}", local_rank)
            min_loss = min(min_loss, test_stats["loss"])
            print_rank_0(f' - Min valid loss: {min_loss:.2f}', local_rank)

            print('- Saving the model after {} epochs ...'.format(epoch))
            save_model(args = args,
                       epoch = epoch,
                       model = model,
                       model_without_ddp = model_without_ddp,
                       optimizer = optimizer,
                       lr_scheduler = lr_scheduler,
                       loss_scaler = loss_scaler,
                       metric = test_stats["loss"],
                       )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print_rank_0(' - Training time {}'.format(total_time_str), local_rank)


if __name__ == "__main__":
    main()

    # pretraining hypers for IN1k
    # base_lr = 0.001
    # min_lr = 0.0
    # batch_size 4096
    # AdamW (beta = [0.9, 0.95], weight_decay = 0.05)
    # patch_size = 16
    # grad_norm = 1.0
    # max_epoch = 4000
    # wp_epoch = 100
    # lr scheduler = cosine
    # 