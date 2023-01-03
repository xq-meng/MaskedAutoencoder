import torch
import torch.nn as nn


class ModelHandler:
    def __init__(
        self,
        model: nn.Module,
    ):
        self.epoch = 0
        self.loss = 1e+8
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(device=self.device)
        self.optimizer = None

    def save_checkpoint(self, path_to_checkpoint):
        torch.save({
            'epoch': self.epoch,
            'loss': self.loss,
            'model_dict': self.model.state_dict(),
            'optimizer_dict': self.optimizer.state_dict() if self.optimizer is not None else None},
            path_to_checkpoint)

    def load_checkpoint(self, path_to_checkpoint, eval=False):
        ckpt = torch.load(path_to_checkpoint)
        self.epoch = ckpt['epoch']
        self.loss = ckpt['loss']
        self.model.load_state_dict(ckpt['model_dict'])
        if (not eval) and (ckpt['optimizer_dict'] is not None) and (self.optimizer is not None):
            self.optimizer.load_state_dict(ckpt['optimizer_dict'])

    def run_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def set_mode(self, mode: str):
        self.model.train(mode == 'train')