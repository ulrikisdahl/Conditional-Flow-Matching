import torch
import ot
import torch.nn as nn
import torch.nn.functional as F
from data.load_data import get_data_loader
from data.load_yosemite import get_yosemite_loader
from modules.u_net import UNet
from modules.vqvae import VQVAE
import torchdiffeq
from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt 
from utils import load_model, display_yosemite



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

def euler_solver(func, y, t, method):
    """
    Simple euler method numerical solver
    """
    h = 1/100
    for t_i in t:
        y = y + h * func(y, t_i)
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


def resample_optimal_plan(source: torch.tensor, target: torch.tensor) -> torch.tensor:
    """
    Mini-batch sampling of optimal transport plan between two distributions 
    Uses euclidian distance measure for cost and Earth Movers Distance for optimal plan
    """ 
    #Assume equal mass  
    source_weights = torch.ones(source.shape[0]) / source.shape[0]
    target_weights = torch.ones(target.shape[0]) / target.shape[0]

    dist = torch.cdist(source.view(source.shape[0], -1), target.view(source.shape[0], -1))**2
    plan = ot.emd(source_weights, target_weights, dist)
    pairs = torch.argwhere(plan > 0)

    #reorder (along the batch dimension only) the target distribution according to corresponding source pairing
    target = target[pairs[:, 1]]
    return source, target



BATCH_SIZE=32
IMG_SIZE=64
EPOCHS=20
lr = 1e-4
device = "cuda"
training=True

data_loader = get_yosemite_loader(
    32, 256, 256,
    path_A="/path/to/domainA",
    path_B="/path/to/domainB",
    split_domains=True
)

#model and optimizers 
flow_model = UNet()
optimizer = torch.optim.Adam(flow_model.parameters(), lr=lr, betas=(0.9, 0.999)) 
flow_model.to(device)
vqvae = load_model(VQVAE, "weights/vqvae_yosemite_best.pth", freeze=True) #load pre-trained VQ-VAE
vqvae.to(device) 
vqvae.eval()

if training: 
    for epoch in range(EPOCHS):
        losses = []
        for batch in tqdm(data_loader):
            """
            1. sample x_0 from source distribution (N(0,1)) and x_1 from target distribution 
            2  re-sample from OT plan
            3. sample x_t using straight probability path 
            4. compute the vector field v
            5. compute static vector field for u
            6. regress
            """     
            optimizer.zero_grad()

            x_0 = batch[0].to(device)
            x_1 = batch[1].to(device)
            
            #create latent representations
            with torch.no_grad():
                z_e_0 = vqvae.encode(x_0, modality="real")
                z_e_1 = vqvae.encode(x_1, modality="real")
                latent_x_0 = vqvae.vector_quantization(z_e_0) #vector quantized latent representation of x_0
                latent_x_1 = vqvae.vector_quantization(z_e_1)

            x_0, x_1 = resample_optimal_plan(x_0, x_1)
            t = torch.rand((BATCH_SIZE, 1))[..., None, None].to(device) #double unsqueeze(-1) such that t can be broadcast
            x_t = t * latent_x_1 + (1 - t) * latent_x_0 #The probability path simpy changes linearly in time
            t = t[:, :, 0, 0] #reshape t for the shape the model expects
            v_t = flow_model(t, x_t)
            loss = static_fm_loss(v_t, latent_x_0, latent_x_1)
            
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, loss: {mean(losses)}") 

    torch.save({
        "model_state_dict": flow_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "weights/latent_flow.pth")
else: 
    #load weights
    checkpoint = torch.load("weights/latent_flow.pth")
    flow_model.load_state_dict(checkpoint["model_state_dict"])


display_yosemite(flow_model, vqvae, data_loader, sample, "cuda")

