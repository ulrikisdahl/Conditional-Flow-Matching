import torch
from data.load_data import get_data_loader
from tqdm import tqdm
from statistics import mean
from modules.vq_vae import VQVAE
import matplotlib.pyplot as plt


BATCH_SIZE=32
IMG_SIZE=256
EPOCHS=10
BETA = 0.25
lr = 1e-4
device = "cuda"
training=False

data_loader = get_data_loader("", BATCH_SIZE, IMG_SIZE)

batch = next(iter(data_loader))
print(batch[0].shape)

model = VQVAE()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

if training:
    for epoch in range(EPOCHS):
        losses = []
        for batch in tqdm(data_loader):
            optimizer.zero_grad()
            
            x = batch[0].to(device)

            x_hat, z_e, z_q = model(x, enc_modality="real", dec_modality="real")

            #compute losses
            reconstruction_loss = torch.mean((x - x_hat) ** 2)
            vq_loss = torch.mean((z_e.detach() - z_q) ** 2)
            commitment_loss = torch.mean((z_e - z_q.detach()) ** 2)
            loss = reconstruction_loss + vq_loss + BETA*commitment_loss
            
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}, loss: {mean(losses)}")

    #save weights
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "weights/vqvae.pth")
else:
    #load weights
    checkpoint = torch.load("weights/vqvae.pth")
    model.load_state_dict(checkpoint["model_state_dict"])


batch = next(iter(data_loader))
fig, axr = plt.subplots(5, figsize=(10,10))
preds, z_e, z_q = model(batch[0].to(device), enc_modality="real", dec_modality="real")
print(preds[0].shape)
for idx in range(5):
    img = preds[idx].permute(1, 2, 0).detach()
    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    axr[idx].imshow(img.cpu().numpy())

plt.show()

