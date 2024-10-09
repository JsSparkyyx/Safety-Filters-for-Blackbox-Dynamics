from method.dynamics import GroundTruthiDBFDynamics, GroundTruthSABLASDynamics, GroundTruthDHSDynamics
from method.barriers import InDCBFAttentionBarrier, SABLASBarrier, DiscriminatingHyperplane
from method.trainers import InDCBFTrainer, SABLASTrainer, HyperplaneTrainer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from pathlib import Path
from data.GroundTruth import DeepAccidentDataset
import torch
import argparse
import os

def main(args):
    tb_logger = TensorBoardLogger(save_dir=args['save_path'],name=args['name'])
    data = DeepAccidentDataset(**args, pin_memory=True)
    if args['method'] == 'InDCBF':
        barrier = InDCBFAttentionBarrier(2,**args)
        model = GroundTruthiDBFDynamics(2,args['device'],latent_dim=args['latent_dim'])
        trainer = InDCBFTrainer(model,barrier,**args)
    elif args['method'] == 'SABLAS':
        barrier = SABLASBarrier(2,**args)
        model = GroundTruthSABLASDynamics(2,args['device'],latent_dim=args['latent_dim'])
        trainer = SABLASTrainer(model,barrier,**args)
    elif args['method'] == 'DH':
        barrier = DiscriminatingHyperplane(2,**args)
        model = GroundTruthDHSDynamics(2,args['device'],latent_dim=args['latent_dim'])
        trainer = HyperplaneTrainer(model,barrier,**args)
    runner = Trainer(logger=tb_logger,
                 callbacks=[
                     ModelCheckpoint(save_top_k=0, 
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
    # parser.add_argument('--backbone',  '-b', default="vc1")
    parser.add_argument('--method',  '-m', default="InDCBF")
    parser.add_argument('--learning_rate',  '-lr', default=0.001)
    parser.add_argument('--weight_decay',  '-wd', default=0)
    parser.add_argument('--w_barrier',  '-wb', default=1)
    parser.add_argument('--w_latent',  '-wl', default=3)
    parser.add_argument('--w_dyn',  '-wdy', default=1)
    parser.add_argument('--w_recon',  '-wr', default=1)
    parser.add_argument('--w_safe', default=1)
    parser.add_argument('--w_unsafe', default=1)
    parser.add_argument('--w_grad', default=1)
    parser.add_argument('--eps_safe', default=0.1)
    parser.add_argument('--eps_unsafe', default=0.1)
    parser.add_argument('--eps_ascent', default=0.1)
    parser.add_argument('--eps_descent', default=0.1)
    parser.add_argument('--gamma_pos', default=1)
    parser.add_argument('--gamma_neg', default=5)
    parser.add_argument('--dt',  '-dt', default=0.1)
    parser.add_argument('--rtol',  '-rtol', default=5e-6)
    parser.add_argument('--window_size',  '-ws', default=2)
    parser.add_argument('--seed',  '-s', default=42)
    parser.add_argument('--train_batch_size', default=256)
    parser.add_argument('--val_batch_size', default=128)
    parser.add_argument('--num_workers', default=16)
    parser.add_argument('--train_barrier', default=True)
    parser.add_argument('--with_dynamic', default=True)
    parser.add_argument('--max_epochs',  '-epoch', default=100)
    parser.add_argument('--latent_dim', default=16)
    parser.add_argument('--save_path',  '-sp', default="/storage1/sibai/Active/yuxuan/tf-logs")
    parser.add_argument('--name', default="GroundTruth")
    args = parser.parse_args()._get_kwargs()
    args = {k:v for (k,v) in args}
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['split_trajectory'] = True
    args['name'] += f"_{args['method']}_{args['seed']}"
    if args['method'] == 'SABLAS':
        args['eps_safe'] = 0.1
        args['eps_unsafe'] = 0.1
        args['eps_ascent'] = 0.0001
        args['w_grad'] = 0.5
    seed_everything(args['seed'])
    main(args)