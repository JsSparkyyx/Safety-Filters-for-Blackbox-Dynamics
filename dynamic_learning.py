from method.inDCBF import InDCBFTrainer, InDCBFController
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from pathlib import Path
from dataset import RACCARDataset
import torch
import argparse
import os

def main(args):
    tb_logger =  TensorBoardLogger(save_dir=args['save_path'],name=args['name'])
    data = RACCARDataset(**args, pin_memory=True)
    model = InDCBFController(1,2,None,args['device'],latent_dim=args['latent_dim'])
    trainer = InDCBFTrainer(model,**args)
    runner = Trainer(logger=tb_logger,
                 callbacks=[
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 max_epochs=args['max_epochs'],
                 gpus = [0])
    
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/ReconDynamic").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/ReconDecode").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Latent").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/LatentDynamic").mkdir(exist_ok=True, parents=True)
    runner.fit(trainer, datamodule=data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate',  '-lr', default=0.001)
    parser.add_argument('--weight_decay',  '-wd', default=0)
    parser.add_argument('--w_latent',  '-wl', default=5)
    parser.add_argument('--w_dyn',  '-wdy', default=5)
    parser.add_argument('--w_recon',  '-wr', default=0.5)
    parser.add_argument('--dt',  '-dt', default=0.05)
    parser.add_argument('--rtol',  '-rtol', default=5e-6)
    parser.add_argument('--window_size',  '-ws', default=10)
    parser.add_argument('--seed',  '-s', default=42)
    parser.add_argument('--train_batch_size', default=32)
    parser.add_argument('--val_batch_size', default=16)
    parser.add_argument('--num_workers', default=4)
    parser.add_argument('--max_epochs',  '-epoch', default=100)
    parser.add_argument('--latent_dim', default=128)
    parser.add_argument('--data_path',  '-tp', default="./trajectories/0")
    parser.add_argument('--save_path',  '-sp', default="./logs/")
    parser.add_argument('--name', default="DynamicLearning")
    args = parser.parse_args()._get_kwargs()
    args = {k:v for (k,v) in args}
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['split_trajectory'] = True
    seed_everything(args['seed'])
    main(args)