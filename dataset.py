from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from torchvision import transforms
import os
import numpy as np
import torch

class RACCAR(Dataset):
    def __init__(self,data_path,split,split_trajectory=False,**kwargs):
        imgs = np.load(os.path.join(data_path,'state.npy'))
        self.imgs = imgs[:int(len(imgs) * 0.9)] if split == "train" else imgs[int(len(imgs) * 0.9):]
        self.us = np.loadtxt(os.path.join(data_path,'velocity.txt'))
        if split_trajectory:
            N,C,H,W = self.imgs.shape
            T = N // 100
            self.imgs = self.imgs[:T*100].reshape(T,100,C,H,W)
            self.us = self.us[:T*100].reshape(T,100,-1)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = img.reshape(img.shape[0],1,img.shape[1],img.shape[2])
        u = self.us[idx] if idx == 0 else self.us[idx-1]
        return torch.Tensor(img), torch.Tensor(u) # dummy datat to prevent breaking 

class RACCARDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        split_trajectory: bool = False,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.split_trajectory = split_trajectory

    def setup(self, stage=None):

        self.train_dataset = RACCAR(
            self.data_dir,
            split='train',
            split_trajectory=self.split_trajectory
        )
        
        self.val_dataset = RACCAR(
            self.data_dir,
            split='val',
            split_trajectory=self.split_trajectory
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )