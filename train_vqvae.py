import torch
from data.load_data import get_data_loader
from data.load_cityscapes import load_cityscapes
from data.load_yosemite import get_yosemite_loader
from tqdm import tqdm
from statistics import mean
# from modules.vq_vae import VQVAE
from modules.vqvae import VQVAE
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity

#####
BATCH_SIZE=32
IMG_SIZE=256
EPOCHS=8
BETA = 0.25
lr = 1e-3
device = "cuda"
training=True
dataset="yosemite" 

#####
if dataset == "cityscapes":
    print("RUNNING CITYSCAPES")
    data_loader = load_cityscapes(
        BATCH_SIZE, 128, 256,
        real_path="/home/ulrik/datasets/cityscapes/full_dataset_kaggle/train/image/",
        seg_path="/home/ulrik/datasets/cityscapes/full_dataset_kaggle/train/label/"
    )
elif dataset == "yosemite":
    print("RUNNING YOSEMITE")
    data_loader = get_yosemite_loader(
        32, 256, 256,
        path_A="/home/ulrik/datasets/yosemite_translation/trainA",
        path_B="/home/ulrik/datasets/yosemite_translation/trainB",
        split_domains=False
    )
else: 
    data_loader = get_data_loader("", BATCH_SIZE, IMG_SIZE)

def perceptual_loss_function(input_image, generated_image):
    """
    Perceptual loss - computes the distance between the input image and the generated image using a pretrained VGG16 model
    """
    input_image_adjusted = (input_image + 1) / 2
    generated_image_adjusted = (generated_image + 1) / 2

    perceptual_loss = learned_perceptual_image_patch_similarity(generated_image_adjusted, input_image_adjusted, normalize=True)
    return perceptual_loss

#####
model = VQVAE()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

if training:
    for epoch in range(EPOCHS):
        losses = []
        for batch in tqdm(data_loader):
            ### forward real modality ###
            optimizer.zero_grad()            
            real_x = batch.to(device)
            x_hat, z_e, z_q = model(real_x, enc_modality="real", dec_modality="real")

            # real_reconstruction_loss = torch.mean((real_x - x_hat) ** 2) 
            real_reconstruction_loss = perceptual_loss_function(real_x, x_hat)
            real_vq_loss = torch.mean((z_e.detach() - z_q) ** 2) 
            real_commitment_loss = torch.mean((z_e - z_q.detach()) ** 2) 
            loss = real_reconstruction_loss + real_vq_loss + BETA*real_commitment_loss
            
            losses.append(loss.item())
            loss.backward()
            optimizer.step()


        print(f"Epoch: {epoch}, loss: {mean(losses)}")

    #save weights
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "weights/vqvae_yosemite.pth")
    print("Saved weights!")
else:
    #load weights
    checkpoint = torch.load("weights/vqvae_yosemite.pth")
    model.load_state_dict(checkpoint["model_state_dict"])



