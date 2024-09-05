import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """
        Creates a time embedding vector for the current time step. Not really a embedding, but rather an encoding used as input for embedding layer
        Args:
            embedding_dim: dimension of the embedding vector
        Returns:
            batch of embedding vectors of shape [batch_size, embedding_dim]
    """
    def __init__(self, embedding_dim):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimensions must be divisible by 2")
        
        self.embedding_dim = embedding_dim

    def forward(self, time_t: torch.tensor) -> torch.tensor:
        """
            time_t: current time step for sample. shape of [batch_size, 1]
        """
        device = time_t.device
        embeddings = torch.zeros((time_t.shape[0], self.embedding_dim), device=device)

        position = torch.arange(0, self.embedding_dim, dtype=torch.float32, device=device)
        div_term = torch.pow(10000, (2 * position // 2) / self.embedding_dim)

        #sine encoding for even indices
        embeddings[:, 0::2] = torch.sin(time_t / div_term[0::2])
        
        #cosine encoding for odd indices
        embeddings[:, 1::2] = torch.cos(time_t / div_term[1::2])
        return embeddings


class Conv2DBlock(nn.Module):
    def __init__(self, input_channels, output_channels, t_dimension, upsample):
        """
        Simple convolutional block with two conv layers (optionally 3) that embeds time
        """
        super(Conv2DBlock, self).__init__()
        self.upsample = upsample

        #Projects the time embedding to fit the dimensions of the input
        self.project_t = nn.Linear(in_features=t_dimension, out_features=output_channels)

        if self.upsample: 
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.ConvTranspose2d(in_channels=output_channels, out_channels=output_channels // 2, kernel_size=2, stride=2)
            self.bn3 = nn.BatchNorm2d(output_channels // 2)
        else:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels) #so far input_c = output_c
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x: torch.tensor, time_embedding: torch.tensor) -> torch.tensor:
        projected_time_embedding = self.project_t(time_embedding)
        projected_time_embedding = torch.relu(projected_time_embedding)
        projected_time_embedding = projected_time_embedding[..., None, None] #unsqueeze two final dimensions
        
        x = torch.relu(self.bn1(self.conv1(x)))
        
        x = x + projected_time_embedding #embed information about the timstep

        x = torch.relu(self.bn2(self.conv2(x)))
        if self.upsample:
            x = torch.relu(self.bn3(self.conv3(x)))
        return x


class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()
        self.t_dimension = 30
        self.channel_dimensions = [3, 64, 128, 256, 512] #[C, 2*C, 3*C, 4*C]


        self.embedding_layer = nn.Sequential(
            SinusoidalEmbedding(self.t_dimension),
            nn.Linear(in_features=self.t_dimension , out_features=self.t_dimension),
            nn.ReLU()
        )

        self.downsample_blocks = nn.ModuleList([])
        for channel_idx in range(len(self.channel_dimensions) - 1):
            self.downsample_blocks.append(
                Conv2DBlock(
                    input_channels=self.channel_dimensions[channel_idx],
                    output_channels=self.channel_dimensions[channel_idx + 1],
                    t_dimension=self.t_dimension,
                    upsample=False
                )
            )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle_conv_block = Conv2DBlock(
            input_channels=self.channel_dimensions[-1],
            output_channels=self.channel_dimensions[-2] * 2 * 2,
            t_dimension=self.t_dimension,
            upsample=True
        )

        self.upsample_blocks = nn.ModuleList([])
        for channel_idx in range(len(self.channel_dimensions) - 2):
            self.upsample_blocks.append(
                Conv2DBlock(
                    input_channels=self.channel_dimensions[::-1][channel_idx] * 2,
                    output_channels=self.channel_dimensions[::-1][channel_idx],
                    t_dimension=self.t_dimension,
                    upsample=True
                )
            )
    
        self.final_conv_block = Conv2DBlock(
            input_channels=self.channel_dimensions[2],
            output_channels=self.channel_dimensions[1],
            t_dimension=self.t_dimension,
            upsample=False
        )

        self.output_conv = nn.Conv2d(in_channels=self.channel_dimensions[1], out_channels=self.channel_dimensions[0], kernel_size=1)

    def forward(self, x: torch.tensor, t: torch.tensor) -> torch.tensor:
        time_embedding = self.embedding_layer(t)

        skip_connections = []
        for down_block in self.downsample_blocks:
            x = down_block(x, time_embedding)
            skip_connections.append(x)
            x = self.maxpool(x)

        x = self.middle_conv_block(x, time_embedding)
        
        for idx, up_block in enumerate(self.upsample_blocks):
            x = torch.cat((x, skip_connections[-idx - 1]), dim=1)
            x = up_block(x, time_embedding)

        x = torch.cat((x, skip_connections[0]), dim=1)
        x = self.final_conv_block(x, time_embedding)
        x = torch.relu(self.output_conv(x))

        return x


if __name__ == "__main__":
    model = U_Net()

    batch = torch.ones((32, 3, 64, 64))

    t = torch.rand((32, 1))
    forward_pass = model(batch, t)
    print(forward_pass.shape)