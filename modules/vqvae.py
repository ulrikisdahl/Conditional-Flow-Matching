import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.residual_sequence = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.residual_sequence(x)

class Encoder(nn.Module):
    def __init__(self, latent_dim: int, input_channels: int = 3):
        super(Encoder, self).__init__()
        self.encoder_sequence = nn.Sequential(
            ResBlock(in_channels=3, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels=64, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels=128, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=latent_dim, kernel_size=1, stride=1) #pointwise
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.encoder_sequence(x)
    
class Decoder(nn.Module):
    def __init__(self, latent_channels, output_channels=3):
        super(Decoder, self).__init__()
        self.initial_conv = nn.Conv2d(latent_channels, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.block1 = ResBlock(256, 256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.block2 = ResBlock(128, 128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.block3 = ResBlock(64, 64)
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.leaky_relu(self.bn1(self.initial_conv(x)))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.block1(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.block2(x)
        
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.leaky_relu(self.bn3(self.conv3(x))) 
        x = self.block3(x)
        x = self.final_conv(x)
        return torch.tanh(x)

class VQVAE(nn.Module):
    """
    Multimodal VQ-VAE that can sample from two different modalities (real images / segmentation maps) with the same codebook
    """
    def __init__(self):
        super(VQVAE, self).__init__()
        self.num_codes=512 * 4
        self.latent_dim=5
        self.num_classes = 19

        self.real_encoder = Encoder(latent_dim=self.latent_dim, input_channels=3)
        self.seg_encoder = Encoder(latent_dim=self.latent_dim, input_channels=1)
        self.real_decoder = Decoder(latent_channels=self.latent_dim, output_channels=3)
        self.seg_decoder = Decoder(latent_channels=self.latent_dim, output_channels=self.num_classes)
        
        self.codebook = nn.Embedding(num_embeddings=512, embedding_dim=self.latent_dim)  

    def encode(self, x: torch.tensor, modality: str) -> torch.tensor: 
        if modality == "real":
            z_e = self.real_encoder(x)        
        else:
            z_e = self.seg_encoder(x)
        return z_e

    def decode(self, z_q: torch.tensor, modality: str) -> torch.tensor:
        if modality == "real":
            x = self.real_decoder(z_q)
        else:
            x = self.seg_decoder(z_q) 
        return x

    def vector_quantization(self, z_e: torch.tensor) -> torch.tensor: 
        """
        Args:
            z_e: Encoded latent representation of shape (B, C, H, W)
        """
        N, C, H, W = z_e.shape
        z_e = z_e.permute(0, 2, 3, 1).view(N, -1, C) #(B, H*W, C)
        dist = torch.cdist(z_e, self.codebook.weight) #distance for each vec in H*w to each code in codebook
        lookup = torch.argmin(dist, dim=-1) #find the closest code for each vec in H*W
        z_q = self.codebook(lookup).view(N, H, W, C).permute(0, 3, 1, 2) #use code to lookup the corresponding vector in the codebook

        return z_q

    def forward(self, x: torch.tensor, enc_modality: str, dec_modality) -> tuple[torch.tensor]:
        z_e = self.encode(x, modality=enc_modality)

        #vector quantization
        z_q = self.vector_quantization(z_e) 
        z_q_ste = z_e + (z_q - z_e).detach() #straight through estimation of gradients through VQ-pass

        output = self.decode(z_q_ste, modality=dec_modality)

        return output, z_e, z_q
    
    
if __name__ == "__main__":
    model = VQVAE()

    x = torch.ones((32, 3, 256, 256))

    x_hat = model(x, enc_modality="real", dec_modality="real")
    print(x_hat[0].shape)
