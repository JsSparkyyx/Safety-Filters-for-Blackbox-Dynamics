from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

DATA_PATH = "/root/autodl-tmp/data/DeepAccident_data"
CAMERA_LOC = ["Camera_Back","Camera_BackLeft","Camera_BackRight","Camera_Front","Camera_FrontLeft","Camera_FrontRight"]
# CAMERA_LOC = ["Camera_Back","Camera_BackLeft"]

class DeepAccident(Dataset):
    def __init__(self,split,**kwargs):
        self.trajectory_list = np.loadtxt(os.path.join(DATA_PATH,f'{split}.txt'),dtype=str)
        
    def __len__(self):
        return len(self.trajectory_list)
    
    def __getitem__(self, idx):
        trajectory = self.trajectory_list[idx]
        camera_path = os.path.join(DATA_PATH,trajectory[0],'ego_vehicle/Camera_Front',trajectory[1])
        label_path = os.path.join(DATA_PATH,trajectory[0],'ego_vehicle/label',trajectory[1])
        transform = transforms.Compose([
            transforms.Resize((224, 224)),   
            transforms.ToTensor(),              
        ])
        label = 0
        if "accident" in trajectory[0]:
            label = 1
        imgs = []
        u = []
        for i in range(1,101):
            img_name = "{}_{:03d}.jpg".format(trajectory[1],i)
            info_name = "{}_{:03d}.txt".format(trajectory[1],i)
            img_path = os.path.join(camera_path,img_name)
            info_path = os.path.join(label_path,info_name)
            img = Image.open(img_path).convert('RGB')
            with open(info_path,'r') as f:
                action = f.readline().strip().split(" ")
            u.append(torch.Tensor([float(action[0]),float(action[1])]))
            sample = transform(img)
            imgs.append(sample)
        return torch.stack(imgs,dim=0), torch.stack(u,dim=0), torch.LongTensor([label])

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
        train_batch_size: int = 2,
        val_batch_size: int = 2,
        test_batch_size: int = 2,
        num_workers: int = 8,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):

        self.train_dataset = DeepAccident(
            split='train',
        )
        
        self.val_dataset = DeepAccident(
            split='val',
        )

        self.test_dataset = DeepAccident(
            split='test',
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
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )