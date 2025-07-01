import sys
import math
import torch

from utils import distributed_utils
from utils.misc import MetricLogger, SmoothedValue
from utils.misc import print_rank_0, all_reduce_mean


def train_one_epoch(args,
                    device,
                    model,
                    data_loader,
                    epoch,
                    optimizer,
                    lr_scheduler,
                    lr_scheduler_warmup,
                    loss_scaler,
                    local_rank,
                    tblogger=None):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    epoch_size = len(data_loader)

    # train one epoch
    for iter_i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        global_steps = iter_i + epoch * epoch_size
        nw = args.wp_epoch * epoch_size // args.update_freq

        # Warmup
        if nw > 0 and global_steps // args.update_freq < nw and global_steps % args.update_freq == 0:
            lr_scheduler_warmup(global_steps // args.update_freq, optimizer)
        elif global_steps // args.update_freq == nw:
            print(" ! Warmup stage is over.")
            lr_scheduler_warmup.set_lr(optimizer, args.base_lr)

        # To device
        images = samples["images"].to(device, non_blocking=True)
        image_prefix_masks = samples["image_prefix_masks"].to(device, non_blocking=True)
        input_ids = samples["input_ids"].to(device, non_blocking=True)
        attention_masks = samples["attention_masks"].to(device, non_blocking=True)

        # Inference
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            ## forward
            loss_dict = model(
                images = images,
                image_prefix_masks = image_prefix_masks,
                input_ids = input_ids,
                attention_mask = attention_masks,
                )
            loss = loss_dict["loss"]
            loss /= args.update_freq


        # Backward & Optimize
        update_grad = (global_steps % args.update_freq == 0)
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=update_grad)
        if update_grad:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Logs
        loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(**loss_dict_reduced)
        metric_logger.update(lr=lr)

        # Check loss
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(" ! Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_value_reduce = all_reduce_mean(loss_value)
        if tblogger is not None and (iter_i + 1) % args.grad_accumulate == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((iter_i / len(data_loader) + epoch) * 1000)
            tblogger.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            tblogger.add_scalar('lr', lr, epoch_1000x)

        # perform per iteration lr schedule
        if global_steps // args.update_freq > nw and update_grad:
            lr_scheduler.step()
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_rank_0(" - Averaged stats: {}".format(metric_logger), local_rank)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, local_rank):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for samples in metric_logger.log_every(data_loader, 10, header):
        # To device
        images = samples["images"].to(device, non_blocking=True)
        image_prefix_masks = samples["image_prefix_masks"].to(device, non_blocking=True)
        input_ids = samples["input_ids"].to(device, non_blocking=True)
        attention_masks = samples["attention_masks"].to(device, non_blocking=True)

        # compute output
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            loss_dict = model(
                images = images,
                image_prefix_masks = image_prefix_masks,
                input_ids = input_ids,
                attention_mask = attention_masks,
                compute_loss = True
                )

        metric_logger.update(**loss_dict)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_rank_0('* valid loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss), local_rank)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
