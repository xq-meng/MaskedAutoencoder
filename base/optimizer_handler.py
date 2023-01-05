import math
import torch.optim


class OptimizerHandler:
    def __init__(
        self,
        parameters,
        optimizer=None,
        name=None,
        optim_params=None,
        lr = -1,
        min_lr = 0.,
        warmup_epochs=-1,
    ):
        # define learning rate into config
        if lr > 0:
            optim_params['lr'] = lr
        # optimizer initialization
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            optim_class = getattr(torch.optim, name)
            self.optimizer = optim_class(params=parameters, **optim_params)
        # parameters for optimizer scheduler 
        self.warmup_epochs = warmup_epochs 
        self.base_lr = optim_params['lr']
        self.min_lr = min_lr

    def update_lr(self, epoch, target_epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (target_epoch - self.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr * param_group["lr_scale"] if "lr_scale" in param_group else lr
        return lr

    def learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]