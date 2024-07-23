from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
import pandas as pd
from torch.utils.data.sampler import WeightedRandomSampler

DATA_PATH = "/root/autodl-tmp/data/DeepAccident_data"
CAMERA_LOC = ["Camera_Back","Camera_BackLeft","Camera_BackRight","Camera_Front","Camera_FrontLeft","Camera_FrontRight"]

class DeepAccident(Dataset):
    def __init__(self,split,**kwargs):
        self.trajectory_list = pd.read_csv(os.path.join(DATA_PATH,f'{split}_label_15.txt'),delimiter=" ")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),     
        ])
        
    def __len__(self):
        return len(self.trajectory_list)
    
    def __getitem__(self, idx):
        trajectory = self.trajectory_list.loc[idx]
        label_path = os.path.join(DATA_PATH,trajectory["type"],trajectory['agent'],'label',trajectory["trajectory"])
        label = trajectory["collision"]
        imgs_all = []
        u = []
        for i in range(trajectory['start'],trajectory['end']+1):
            info_name = "{}_{:03d}.txt".format(trajectory['trajectory'],i)
            info_path = os.path.join(label_path,info_name)
            with open(info_path,'r') as f:
                action = f.readline().strip().split(" ")
            u.append(torch.Tensor([float(action[0]),float(action[1])]))
            imgs = []
            for cam_idx in CAMERA_LOC:
                camera_path = os.path.join(DATA_PATH,trajectory['type'],trajectory['agent'],cam_idx,trajectory['trajectory'])
                img_name = "{}_{:03d}.jpg".format(trajectory['trajectory'],i)
                img_path = os.path.join(camera_path,img_name)
                img = read_image(img_path)/255
                sample = self.transform(img)
                imgs.append(sample)
            imgs = torch.stack(imgs,dim=0)
            imgs_all.append(imgs)
        return torch.stack(imgs_all,dim=0), torch.stack(u,dim=0), torch.LongTensor([label])

class DeepAccidentDataset(LightningDataModule):
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
        train_batch_size: int = 16,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def setup(self, stage=None):

        self.train_dataset = DeepAccident(
            split='train',
        )
        
        self.val_dataset = DeepAccident(
            split='val',
        )

        # self.test_dataset = DeepAccident(
        #     split='test',
        # )
        
    def train_dataloader(self) -> DataLoader:
        # weights = [20 if label[0] == 1 else 1 for i, u, label in self.train_dataset]
        # sampler = WeightedRandomSampler(weights,num_samples=len(weights), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            # sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )