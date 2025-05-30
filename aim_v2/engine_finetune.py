import sys
import math
import numpy as np

import torch

from utils.misc import MetricLogger, SmoothedValue
from utils.misc import print_rank_0, all_reduce_mean, accuracy


def train_one_epoch(args,
                    device,
                    model,
                    data_loader,
                    epoch,
                    optimizer,
                    lr_scheduler,
                    lr_scheduler_warmup,
                    loss_scaler,
                    criterion,
                    local_rank=0,
                    tblogger=None,
                    mixup_fn=None):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    epoch_size = len(data_loader)

    optimizer.zero_grad()

    # train one epoch
    for iter_i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ni = iter_i + epoch * epoch_size
        nw = lr_scheduler_warmup.wp_iter
        
        # Warmup
        if nw > 0 and ni < nw:
            lr_scheduler_warmup(ni, optimizer)
        elif ni == nw:
            print("Warmup stage is over.")
            lr_scheduler_warmup.set_lr(optimizer, args.base_lr)

        # To device
        images = samples[0].to(device, non_blocking=True)
        targets = samples[-1].to(device, non_blocking=True).long()

        # Mixup
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        # Inference
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, targets)

        # Check loss
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Backward & Optimize
        loss_scaler(loss = loss,
                    optimizer = optimizer,
                    clip_grad = args.max_grad_norm, 
                    parameters = model.parameters(),
                    create_graph = False,
                    )
        optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Logs
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)

        loss_value_reduce = all_reduce_mean(loss_value)
        if tblogger is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((iter_i / len(data_loader) + epoch) * 1000)
            tblogger.add_scalar('loss', loss_value_reduce, epoch_1000x)
            tblogger.add_scalar('lr', lr, epoch_1000x)

        # perform per iteration lr schedule
        if ni > nw:
            lr_scheduler.step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_rank_0("Averaged stats: {}".format(metric_logger), local_rank)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, local_rank):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for samples in metric_logger.log_every(data_loader, 10, header):
        images = samples[0]
        target = samples[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_rank_0('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                 .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss),
                 local_rank)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
