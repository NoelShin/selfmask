from torch.optim.lr_scheduler import _LRScheduler


class Poly(_LRScheduler):
    def __init__(
            self,
            optimizer,
            n_epochs: int,
            n_iters_per_epoch: int,
            warmup_epochs: int = 0,
            warmup_iters: int = None,
            last_epoch: int = -1
    ):
        self.n_iters_per_epoch = n_iters_per_epoch
        self.cur_iter = 0
        self.total_iter = n_epochs * n_iters_per_epoch
        self.warmup_iters = warmup_epochs * n_iters_per_epoch if warmup_iters is None else warmup_iters
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        # noel - as Poly learning rate does NOT care about epoch but the total number of steps, I just ignore
        # self.last_epoch
        T = self.cur_iter

        if 0 < self.warmup_iters and T < self.warmup_iters:
            # noel - linearly increase learning rate from 0 to base lr.
            factor = 1.0 * T / self.warmup_iters

        else:
            factor = pow((1 - 1.0 * T / self.total_iter), 0.9)

        self.cur_iter %= self.n_iters_per_epoch
        self.cur_iter += 1
        assert factor >= 0, 'error in lr_scheduler'
        return [base_lr * factor for base_lr in self.base_lrs]