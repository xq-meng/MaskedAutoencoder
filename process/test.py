import os
import torch
import torch.nn as nn
import base
import utils


class TestHandler(base.ModelHandler):
    def __init__(
        self,
        model: nn.Module,
        data_loader,
        checkpoint_path=None,
        output_dir=None,
        logger=None,
        **kwargs
    ):
        super(TestHandler, self).__init__(model)
        # testing parameters
        self.output_dir = output_dir
        utils.mkdir(self.output_dir)
        # data
        self.data_loader = data_loader
        # model setup
        self.model = model
        if checkpoint_path is not None and os.access(checkpoint_path, os.R_OK):
            self.load_checkpoint(checkpoint_path)
        self.model.to(device=self.device)
        self.model.eval()
        # logger
        self.logger = logger if logger is not None else utils.Logger()
        # kwargs
        self.kwargs = kwargs

    def run(self):
        for step, datas in enumerate(self.data_loader):
            x = datas['image'].to(self.device)
            with torch.no_grad():
                ret = self.model.inference(x, **self.kwargs)
            imgs = utils.tensor2PIL(ret)
            for i, filename in enumerate(datas['name']):
                output_path = os.path.join(self.output_dir, filename)
                imgs[i].save(output_path)
                self.logger.info("Test output save as {}".format(output_path))
