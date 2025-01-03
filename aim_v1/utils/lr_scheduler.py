# Linear Warmup Scheduler
class LinearWarmUpLrScheduler(object):
    def __init__(self, base_lr=0.01, wp_iter=500, warmup_factor=0.00066667):
        self.base_lr = base_lr
        self.wp_iter = wp_iter
        self.warmup_factor = warmup_factor

    def set_lr(self, optimizer, cur_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr

    def __call__(self, iter, optimizer):
        # warmup
        assert iter < self.wp_iter
        alpha = iter / self.wp_iter
        warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        tmp_lr = self.base_lr * warmup_factor
        self.set_lr(optimizer, tmp_lr)
