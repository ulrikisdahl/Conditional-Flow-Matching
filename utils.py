import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.optim import Optimizer
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt

def load_model(
        model: nn.Module,
        checkpoint: str,
        optimizer: Optional[Optimizer] = None,
        freeze: bool = False
    ) -> Union[nn.Module, Tuple[nn.Module, Optimizer]]:
    """
    Loads the model from checpoint and optionally the optimizer if specified
    """
    model = model() 
    checkpoints = torch.load(checkpoint)
    model.load_state_dict(checkpoints["model_state_dict"])
    
    if freeze:
        for layers in model.parameters():
            layers.requires_grad = False
        return model
    
    if optimizer:
        optimizer.load_state_dict(checkpoints["optimizer_state_dict"])
        return (model, optimizer)
    
    return model


def display_yosemite(model: nn.Module, vqvae: nn.Module, data_loader: DataLoader, sampler, device: str) -> None:
    model.eval()
    batch = next(iter(data_loader))

    fig, axr = plt.subplots(3, 2, figsize=(10,10))
    for idx in range(6):
        with torch.no_grad():
            latent_encoding = vqvae.encode(batch[0][idx][None, ...].to(device), modality="real")[0] #take one
            latent_encoding = latent_encoding[None, ...]
            latent_encoding = vqvae.vector_quantization(latent_encoding)

        solution = sampler(model, latent_encoding, 100, "rk4")

        solution = solution[-1]
        with torch.no_grad(): 
            decoded = vqvae.vector_quantization(solution[None, ...])
            decoded = vqvae.decode(decoded, modality="real")
    
        img = decoded[0].permute(1, 2, 0).to("cpu")
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        
        row = idx // 2
        col = idx % 2
        axr[row][col].imshow(img)

    plt.show()


def display_flowers102(model: nn.Module, data_loader: DataLoader, sampler,  device: str) -> None:
    model.eval()
    batch = next(iter(data_loader)) 

    fig, axr = plt.subplots(3, 2, figsize=(10, 10))
    for idx in range(6):
        sample_shape = batch[0][0][None, ...].to(device)
        solution = sampler(model, sample_shape, 100, "rk4")

        img = solution[-1]
        img = img.permute(1, 2, 0).to("cpu")
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))

        row = idx // 2
        col = idx % 2
        axr[row][col].imshow(img)

    plt.show()

