import torch
import torch.nn as nn
import torch.nn.functional as F
from data.load_data import get_data_loader
from modules.u_net import U_Net
import torchdiffeq
from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt 
from modules.test_unet import BasicUNet


def euler_solver(func, y, t, method):
    """
    Simple euler method numerical solver
    """
    h = 1/100
    for t_i in t:
        y = y + h * func(y, t_i)
    return y

def rk4_solver(func, y, t, method):
    """
    Simple fixed-time runge kutta solver
    """
    h = 1/100
    for t_i in t:
        k1 = func(t_i, y)
        k2 = func(t_i + h*0.5, y + 0.5*h*k1)
        k3 = func(t_i + h*0.5, y + 0.5*h*k2)
        k4 = func(t_i + h, y + h*k3)
        y = y + h*(k1 + 2*k2 + 2*k3 + k4)/6         
    return y


def sample(model: nn.Module, data_sample: torch.tensor, num_steps: int, ode_method: str):
    """
    Samples the ODE using the learned vector field
    """
    x_0 = torch.randn_like(data_sample).to(data_sample.device)

    if ode_method == "dopri5":
        solver = torchdiffeq.odeint        
        t = torch.linspace(0, 1, 2).to(data_sample.device)
    elif ode_method == "euler":
        solver = euler_solver
        t = torch.linspace(1e-2, 1, num_steps).to(data_sample.device)
    elif ode_method == "rk4":
        t = torch.linspace(1e-2, 1, num_steps).to(data_sample.device)
        solver = rk4_solver
        
    with torch.no_grad():
        solution = solver(model, x_0, t, method=ode_method)

    if ode_method == "dopri5":
        solution = solution[:, 0, :, :, :]
    return solution

def static_fm_loss(predicted_vector_field: torch.tensor, source_distibution: torch.tensor, target_distribution: torch.tensor) -> torch.tensor:
    """
    Regresses the predicted vectorfield to the target vectorfield
    """
    target_vector_field = (target_distribution - source_distibution)
    return F.mse_loss(target_vector_field, predicted_vector_field) 


BATCH_SIZE=32
IMG_SIZE=64
EPOCHS=20
lr = 1e-4
device = "cuda"
training=False

data_loader = get_data_loader("", BATCH_SIZE, IMG_SIZE)
batch = next(iter(data_loader))
print(batch[0].shape)

#model and optimizers 
# model = U_Net()
model = BasicUNet()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999)) 
model.to(device)




if training: 
    for epoch in range(EPOCHS):
        losses = []
        for batch in tqdm(data_loader):
            """
            1. sample x_0 from source distribution (N(0,1)) and x_1 (target) from target distribution 
            1.5 optinally: re-sample from OT
            2. sample x_t using straight probability path 
            3. compute the vector field v
            4. compute vector field for u
            5. regress
            """     
            optimizer.zero_grad()

            target = batch[0].to(device)
            x_0 = torch.randn_like(target).to(device)
            t = torch.rand((BATCH_SIZE, 1))[..., None, None].to(device) #double unsqueeze(-1) such that t can be broadcast
            x_t = t * target + (1 - t) * x_0 #The probability path simpy changes linearly in time
            t = t[:, :, 0, 0] #resize t for the shape the model expects
            v_t = model(t, x_t)
            loss = static_fm_loss(v_t, x_0, target)
            
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, loss: {mean(losses)}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "weights/first.pth")
else: 
    #load weights
    checkpoint = torch.load("weights/first.pth")
    model.load_state_dict(checkpoint["model_state_dict"])





if __name__ == "__main__":
    model.eval()

    fig, axr = plt.subplots(3, 2, figsize=(10, 10))
    for idx in range(6):
        batch = next(iter(data_loader))[0][0][None, ...].to(device) #unsqueeze at the end
        solution = sample(model, batch, 100, "dopri5")

        img = solution[-1]
        img = img.permute(1, 2, 0).to("cpu")
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        
        row = idx // 2
        col = idx % 2
        axr[row][col].imshow(img)
        print(torch.min(img), torch.max(img))

    plt.show()

#Throughts
# - Both papers seem to use dopri5 as an ODE solver
# - IMPORTANT: During training we dont solve the ODEs, so we never actually do anything beyond the vector field, but during sampling we must solve ODEs using the learned vector field!
# - C = 128, [C, 2C, 3C, 4C] in the downsampling




