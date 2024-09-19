"""
Dataloader for the cityscapes segmentation dataset: https://www.cityscapes-dataset.com/
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data as td
import torchvision
import os


class CityScapesDataset(td.Dataset):
    """
    args
        real_path: path to dataset of real images
        seg_path: path to segmentation dataset
        preload: load the entire datset into memory upfront for faster training  
    """
    def __init__(
            self,
            transforms: torchvision.transforms, 
            real_path: str, 
            seg_path: str, 
            preload=False
        ):
        super(CityScapesDataset, self).__init__()
        self.preload = preload
        self.img_transforms = transforms
        self.length = len(os.listdir(real_path))

        if self.preload:
            self.real_images = []
            for img in Path(real_path).iterdir():
                np_img = np.load(img)
                assert(len(np_img.shape) == 3)
                img = self.img_transforms(np_img)
                self.real_images.append(img)
            
            self.seg_maps = []
            for seg in Path(seg_path).iterdir():
                np_seg = np.load(seg)
                assert(len(np_seg.shape) == 2)
                seg = self.img_transforms(np_seg)
                self.seg_maps.append(seg)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.preload:
            img = self.real_images[idx]
            seg_map = self.seg_maps[idx]
            return (img, seg_map)
        
        raise NotImplemented("Datset without preloading is not currently implemented")
        return

def load_cityscapes(
        batch_size: int,
        height: int, 
        width: int, 
        real_path: str, 
        seg_path: str) -> td.DataLoader:
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((height, width)),
        torchvision.transforms.Lambda(lambda x: x.float()),
        torchvision.transforms.Lambda(lambda x: (x * 2) - 1)
    ])

    dataset = CityScapesDataset(
        transforms=transforms,
        real_path=real_path,
        seg_path=seg_path,
        preload=True
    )
    print(f"len: {len(dataset)}")

    dataloader = td.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader 



if __name__ == "__main__":

    path_str = "/home/ulrik/datasets/cityscapes/full_dataset_kaggle/train/image/"
    path = Path(path_str)

    print(len(os.listdir(path_str)))

    dataset = CityScapesDataset(
        transforms=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((128, 256)),
            # torchvision.transforms.Lambda(lambda t: (t * 2) - 1),
        ]),
        real_path=path_str,
        seg_path="/home/ulrik/datasets/cityscapes/full_dataset_kaggle/train/label/",
        preload=True
    )

    dataloader = td.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    batch = next(iter(dataloader))

    print(batch[0].shape)
    print(batch[1].shape)

    print(torch.unique(batch[1]))

    #count number of elements in batch[1] equal to -1.0
    print(torch.sum(batch[1] == 2.0))
