import os 
import torch 
import torch.nn as nn
import base
import utils


class TrainHandler(base.ModelHandler):
    def __init__(
        self, 
        model: nn.Module,
        data_loader,
        optimizer_config,
        target_epochs=-1,
        checkpoint_epoch=20,
        checkpoint_dir=None,
        logging_step=200,
        logger=None,
        **kwargs
    ):
        super(TrainHandler, self).__init__(model)
        # data loader
        self.data_loader = data_loader
        # training parameters
        self.target_epochs = target_epochs
        # optimizer
        self.optimizer_handler = base.OptimizerHandler(parameters=self.model.parameters(), **optimizer_config)
        # logger
        self.logger = logger if logger is not None else utils.Logger()
        # logging step
        self.logging_step = logging_step
        # checkpoints
        self.checkpoint_epoch = checkpoint_epoch if checkpoint_dir is not None else 10000
        self.checkpoint_dir = checkpoint_dir
        utils.mkdir(self.checkpoint_dir)

    def run(self):
        while self.epoch < self.target_epochs:
            for step, datas in enumerate(self.data_loader):
                # zero grad
                self.optimizer_handler.optimizer.zero_grad()
                # update optimizer
                self.optimizer_handler.update_lr(epoch=(step / len(self.data_loader) + self.epoch), target_epoch=self.target_epochs)
                # load data 
                x = datas['image'].to(self.device)
                # model step
                self.loss = self.model(x)
                # optimzier step
                self.loss.backward()
                self.optimizer_handler.optimizer.step()
                # logging training info
                if step % self.logging_step == 0:
                    self.logger.info(f'Epoch = {self.epoch}, LR = {self.optimizer_handler.learning_rate():.4e}, Loss = {self.loss:.4e}')
            if (self.checkpoint_dir is not None) and (self.epoch % self.checkpoint_epoch == 0):
                checkpoint_filename = os.path.join(self.checkpoint_dir, f'ep_{self.epoch}.pkl')
                self.save_checkpoint(checkpoint_filename)
                self.logger.info(f'Epoch = {self.epoch}, Checkpoint saved as: {checkpoint_filename}')
            # update epoch
            self.epoch += 1
        # save final status
        if self.checkpoint_dir is not None:
            self.save_checkpoint(os.path.join(self.checkpoint_dir, "trained.pkl"))
            self.logger.info(f'Epoch = {self.epoch}, Checkpoint saved as: {checkpoint_filename}')