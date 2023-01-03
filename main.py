import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import utils
import masked_autoencoder
from data.anime import AnimeDataset
import process


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    # config
    with open(args.config, 'r') as fr:
        config = json.load(fr)

    # logger 
    logger = utils.Logger(name='base_logger', **config['logging'])

    # model
    model = masked_autoencoder.MAE(**config['model'])

    # train
    if 'train' in config['tasks'] and config['tasks']['train']['run'] == True:
        dataset = AnimeDataset(**config['tasks']['train']['data']['dataset'])
        dataloader = DataLoader(dataset=dataset, **config['tasks']['train']['data']['dataloader'])
        trainer = process.TrainHandler(model=model, data_loader=dataloader, logger=logger, **config['tasks']['train'])
        trainer.run()
    
    # test
    if 'test' in config['tasks'] and config['tasks']['test']['run'] == True:
        dataset = AnimeDataset(**config['tasks']['test']['data']['dataset'])
        dataloader = DataLoader(dataset=dataset, **config['tasks']['test']['data']['dataloader'])
        tester = process.TestHandler(model=model, data_loader=dataloader, logger=logger, **config['tasks']['test'])
        tester.run()