"""
Data loader for flowers-102 dataset
"""

from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import torch

def load_transformed_dataset(img_size: int):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.Flowers102(root=".", download=True, 
                                         transform=data_transform)
    
    print(f"len: {len(train)}")

    test = torchvision.datasets.Flowers102(root=".", download=True, 
                                         transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train, test])


def get_data_loader(path: str, batch_size: int, img_size: int) -> DataLoader:
    """
        Returns the a torch data loader with the images in range [-1, 1]
    """
    data = load_transformed_dataset(img_size) #example dataset
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader