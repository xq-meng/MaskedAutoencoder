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
        target_epochs=-1,
        learning_rate=1e-3,
        optimizer_beta1=0.9,
        optimizer_beta2=0.999,
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(optimizer_beta1, optimizer_beta2))
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
                self.optimizer.zero_grad()
                x = datas['reference'].to(self.device)
                self.loss = self.model(x)
                self.loss.backward()
                self.optimizer.step()
                # logging training info
                if step % self.logging_step == 0:
                    self.logger.info(f'Epoch = {self.epoch}, Loss = {self.loss:.4e}')
            if (self.checkpoint_dir is not None) and (self.epoch % self.checkpoint_epoch == 0):
                checkpoint_filename = os.path.join(self.checkpoint_dir, "ep_{}.pkl".format(self.epoch))
                self.save_checkpoint(checkpoint_filename)
                self.logger.info("Epoch = %d, Checkpoint saved as: %s", self.epoch, checkpoint_filename)
            # update epoch
            self.epoch += 1
        # save final status
        if self.checkpoint_dir is not None:
            self.save_checkpoint(os.path.join(self.checkpoint_dir, "trained.pkl"))