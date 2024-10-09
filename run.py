from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from init_parameters import init_parameters
from pathlib import Path
import importlib
import torch
import argparse
import os

def main(args):
    tb_logger = TensorBoardLogger(save_dir=args['save_path'],name=args['name'])
    method = args['method']
    barriers = importlib.import_module('method.barriers')
    dynamics = importlib.import_module('method.dynamics')
    trainers = importlib.import_module('method.trainers')
    if args['groundtruth']:
        from data.GroundTruth import DeepAccidentDataset
        model = getattr(dynamics, f'GroundTruth{method}Dynamics')(2,args['device'],model=args['backbone'],latent_dim=args['latent_dim'])
    else:
        from data.Safe2Unsafe import DeepAccidentDataset
        model = getattr(dynamics, f'{method}Dynamics')(2,args['device'],model=args['backbone'],latent_dim=args['latent_dim'])
    data = DeepAccidentDataset(**args, pin_memory=True)
    barrier = getattr(barriers, f'{method}Barrier')(2,**args)
    trainer = getattr(trainers,f'{method}Trainer')(model,barrier,**args)
    runner = Trainer(logger=tb_logger,
                 callbacks=[
                     ModelCheckpoint(save_top_k=0, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "loss",
                                     save_last= True),
                 ],
                 max_epochs=args['epoch'],
                 gpus = [0])
    runner.fit(trainer, datamodule=data)

if __name__ == '__main__':
    args = init_parameters()
    seed_everything(args['seed'])
    main(args)