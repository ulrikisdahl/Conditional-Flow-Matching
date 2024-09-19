"""
Data loader for yosemite summer2winter dataset: https://www.kaggle.com/datasets/balraj98/summer2winter-yosemite 
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
from PIL import Image
import random

class YosemiteDataset(td.Dataset):
    """
    args
        preload: load the entire datset into memory upfront for faster training  
    """
    def __init__(
            self,
            transforms: torchvision.transforms, 
            path_A: str, 
            path_B: str,
            split_domains: str, 
            preload=False
        ):
        super(YosemiteDataset, self).__init__()
        self.preload = preload
        self.img_transforms = transforms
        self.length = len(os.listdir(path_A))
        self.path_A = path_A
        self.path_B = path_B
        self.split_domains = split_domains

        if self.preload:
            if split_domains:
                self.domain_A = []
                self.domain_B = []
                self.load_split_domains()
            else:
                self.dataset = []
                self.load_merged_domains()

    def load_merged_domains(self):
        """
        Loads the two domains into the same dataset
        """
        for img in Path(self.path_A).iterdir():
            img = Image.open(img)
            img = self.img_transforms(img)
            self.dataset.append(img)

        for img in Path(self.path_B).iterdir():
            img = Image.open(img)
            img = self.img_transforms(img)
            self.dataset.append(img)

        random.shuffle(self.dataset) #shouldnt need this, but somehow do 
    
    def load_split_domains(self):
        """
        Loads the two domains into seperate datasets for unpaired translation
        """
        for img in Path(self.path_A).iterdir():
            img = Image.open(img)
            img = self.img_transforms(img)
            self.domain_A.append(img)

        for img in Path(self.path_B).iterdir():
            img = Image.open(img)
            img = self.img_transforms(img)
            self.domain_B.append(img)

        domain_A_len = 1540
        domain_B_len = 1200
        #Re-add random elements from domain_B to make it the same length as domain_A
        for i in range(domain_A_len - domain_B_len):
            rand_idx = np.random.randint(0, domain_B_len)
            self.domain_B.append(self.domain_B[rand_idx])

        assert(len(self.domain_A) == len(self.domain_B))
        

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.preload:
            if self.split_domains:
                return (self.domain_A[idx], self.domain_B[idx])
            return self.dataset[idx]
        
        raise NotImplemented("Datset without preloading is not currently implemented")


def get_yosemite_loader(
        batch_size: int,
        height: int, 
        width: int, 
        path_A: str, 
        path_B: str,
        split_domains: str) -> td.DataLoader:
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((height, width)),
        torchvision.transforms.Lambda(lambda x: x.float()),
        torchvision.transforms.Lambda(lambda x: (x * 2) - 1)
    ])

    dataset = YosemiteDataset(
        transforms=transforms,
        path_A=path_A,
        path_B=path_B,
        split_domains=split_domains,
        preload=True
    )
    print(f"len: {len(dataset)}")

    dataloader = td.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader




if __name__ == "__main__":    
    
    #Example
    data_loader = get_yosemite_loader(
        32, 256, 256,
        path_A="/path/to/domainA",
        path_B="/path/to/domainB", 
        split_domains=True
    )

    batch = next(iter(data_loader))
    print(len(batch))
    print(batch[0].shape)
    print(batch[1].shape)




