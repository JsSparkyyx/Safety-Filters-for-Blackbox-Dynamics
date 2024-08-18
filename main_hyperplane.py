from method.dynamics import HyperplaneEncoder
from method.barriers import DiscriminatingHyperplane
from method.trainers import HyperplaneTrainer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from pathlib import Path
from data.Safe2Unsafe import DeepAccidentDataset
import torch
import argparse
import os

def main(args):
    tb_logger = TensorBoardLogger(save_dir=args['save_path'],name=args['name'])
    data = DeepAccidentDataset(**args, pin_memory=True)

    barrier = DiscriminatingHyperplane(2,**args)
    model = HyperplaneEncoder(2,args['device'],model=args['backbone'],latent_dim=args['latent_dim'])
    trainer = HyperplaneTrainer(model,barrier,**args)
    runner = Trainer(logger=tb_logger,
                 callbacks=[
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "loss",
                                     save_last= True),
                 ],
                 max_epochs=args['max_epochs'],
                 gpus = [0])
    runner.fit(trainer, datamodule=data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--backbone',  '-b', default="stabilityai/sd-vae-ft-mse")
    # parser.add_argument('--backbone',  '-b', default="openai/clip-vit-base-patch16")
    parser.add_argument('--backbone',  '-b', default="resnet50")
    # parser.add_argument('--backbone',  '-b', default="google/vit-base-patch16-224")
    parser.add_argument('--learning_rate',  '-lr', default=0.001)
    parser.add_argument('--weight_decay',  '-wd', default=0)
    parser.add_argument('--gamma_pos', default=1)
    parser.add_argument('--gamma_neg', default=5)
    parser.add_argument('--seed',  '-s', default=42)
    parser.add_argument('--train_batch_size', default=32)
    parser.add_argument('--val_batch_size', default=32)
    parser.add_argument('--num_workers', default=16)
    parser.add_argument('--max_epochs',  '-epoch', default=30)
    parser.add_argument('--latent_dim', default=16)
    parser.add_argument('--save_path',  '-sp', default="/root/tf-logs/")
    parser.add_argument('--name', default="DiscriminatingHyperplane")
    args = parser.parse_args()._get_kwargs()
    args = {k:v for (k,v) in args}
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['split_trajectory'] = True
    seed_everything(args['seed'])
    main(args)