import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if time_t.ndimension() == 0:
            time_t = time_t.unsqueeze(0).unsqueeze(0)

        device = time_t.device
        embeddings = torch.zeros((time_t.shape[0], self.embedding_dim), device=device)

        position = torch.arange(0, self.embedding_dim, dtype=torch.float32, device=device)
        div_term = torch.pow(10000, (2 * position // 2) / self.embedding_dim)

        #sine encoding for even indices
        embeddings[:, 0::2] = torch.sin(time_t / div_term[0::2])
        
        #cosine encoding for odd indices
        embeddings[:, 1::2] = torch.cos(time_t / div_term[1::2])
        return embeddings
    

class MultiHeadAttention(nn.Module):
    def __init__(self, channels, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.downsample_after=False #Hack for making compatible with UNet
        self.upsample=False

        # self.group_norm = nn.GroupNorm(num_groups=n_heads, num_channels=channels)
        self.group_norm = nn.BatchNorm2d(channels)

        self.Q_proj = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1) 
        self.K_proj = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.V_proj = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        
        self.out_proj = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        B, C, Height, Width = x.shape

        x = self.group_norm(x)

        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)

        Q = Q.view(B, self.n_heads, self.head_dim, Height*Width).permute(0, 1, 3, 2) #(B, n_heads, H*W, head_dim)
        K = K.view(B, self.n_heads, self.head_dim, Height*Width).permute(0, 1, 3, 2)
        V = V.view(B, self.n_heads, self.head_dim, Height*Width).permute(0, 1, 3, 2)

        attn_scores = torch.einsum("bnhq, bnhk -> bnhk", Q, V) 
        attn_scores = attn_scores / (self.head_dim**0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)

        attn_output = torch.einsum("bnhq, bnhv -> bnhv", attn_scores, V)

        attn_output = attn_output.permute(0, 1, 3, 2).contiguous().view(B, C, Height, Width)
        out = self.out_proj(attn_output)
        return attn_output


class Conv2DBlock(nn.Module):
    def __init__(self, input_channels, output_channels, t_dimension, upsample, downsample_after=False):
        """
        Simple convolutional block with two conv layers (optionally 3) that embeds time
        """
        super(Conv2DBlock, self).__init__()
        self.upsample = upsample
        self.downsample_after = downsample_after

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
        
        h = torch.relu(self.bn1(self.conv1(x)))
        h = h + projected_time_embedding #embed information about the timstep
        h = torch.relu(self.bn2(self.conv2(h)))
        # h = h + x

        if self.upsample:
            h = torch.relu(self.bn3(self.conv3(h)))
        return h


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.t_dimension = 30
        self.latent_dim = 5
        self.channel_dimensions = [self.latent_dim, 64, 128, 256, 512] #[C, 2*C, 3*C, 4*C]


        self.embedding_layer = nn.Sequential(
            SinusoidalEmbedding(self.t_dimension),
            nn.Linear(in_features=self.t_dimension , out_features=self.t_dimension),
            nn.ReLU()
        )

        ### Downsample ###
        self.downsample_blocks = nn.ModuleList([])
        for channel_idx in range(len(self.channel_dimensions) - 1):
            if channel_idx > 1:
                self.downsample_blocks.append(
                    Conv2DBlock(
                        input_channels=self.channel_dimensions[channel_idx],
                        output_channels=self.channel_dimensions[channel_idx + 1],
                        t_dimension=self.t_dimension,
                        upsample=False,
                        downsample_after=False
                    )
                )
                self.downsample_blocks.append(
                    MultiHeadAttention(
                        channels=self.channel_dimensions[channel_idx + 1],
                        n_heads=4
                    )  
                )
                self.downsample_blocks.append(
                    Conv2DBlock(
                        input_channels=self.channel_dimensions[channel_idx + 1],
                        output_channels=self.channel_dimensions[channel_idx + 1],
                        t_dimension=self.t_dimension,
                        upsample=False,
                        downsample_after=True
                    )
                )
            else:
                self.downsample_blocks.append(
                    Conv2DBlock(
                        input_channels=self.channel_dimensions[channel_idx],
                        output_channels=self.channel_dimensions[channel_idx + 1],
                        t_dimension=self.t_dimension,
                        upsample=False,
                        downsample_after=True
                    )
                )
            
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        ### Middle ###
        self.middle_conv_block1 = Conv2DBlock(
            input_channels=self.channel_dimensions[-1],
            output_channels=self.channel_dimensions[-2] * 2 * 2,
            t_dimension=self.t_dimension,
            upsample=False
        )
        self.middle_attention_block1 = MultiHeadAttention(
            channels=self.channel_dimensions[-2] * 2 * 2,
            n_heads=4
        )
        self.middle_conv_block2 = Conv2DBlock(
            input_channels=self.channel_dimensions[-2] * 2 * 2,
            output_channels=self.channel_dimensions[-2] * 2 * 2,
            t_dimension=self.t_dimension,
            upsample=True
        )

        ### Upsample ###
        self.upsample_blocks = nn.ModuleList([])
        for channel_idx in range(len(self.channel_dimensions) - 2):
            if channel_idx < 2:
                self.upsample_blocks.append(
                    Conv2DBlock(
                        input_channels=self.channel_dimensions[::-1][channel_idx] * 2,
                        output_channels=self.channel_dimensions[::-1][channel_idx], #we want to halve it
                        t_dimension=self.t_dimension,
                        upsample=False
                    )
                )
                self.upsample_blocks.append(
                    MultiHeadAttention(
                        channels=self.channel_dimensions[::-1][channel_idx],
                        n_heads=4
                    )
                )
                self.upsample_blocks.append(
                    Conv2DBlock(
                        input_channels=self.channel_dimensions[::-1][channel_idx],
                        output_channels=self.channel_dimensions[::-1][channel_idx], #we want to halve it
                        t_dimension=self.t_dimension,
                        upsample=True
                    )
                )
            else:
                self.upsample_blocks.append(
                    Conv2DBlock(
                        input_channels=self.channel_dimensions[::-1][channel_idx] * 2,
                        output_channels=self.channel_dimensions[::-1][channel_idx], #we want to halve it
                        t_dimension=self.t_dimension,
                        upsample=True
                    )
                )

        self.upsample_blocks.append(
            Conv2DBlock(
                input_channels=self.channel_dimensions[2],
                output_channels=self.channel_dimensions[1],
                t_dimension=self.t_dimension,
                upsample=False
            )            
        )
    
        self.output_conv = nn.Conv2d(in_channels=self.channel_dimensions[1], out_channels=self.channel_dimensions[0], kernel_size=1)

    def forward(self, t: torch.tensor, x: torch.tensor) -> torch.tensor:
        """
        Has to take in time argument first because of torchdiffeq solver
        """
        time_embedding = self.embedding_layer(t)

        skip_connections = []
        for down_block in self.downsample_blocks:
            x = down_block(x, time_embedding)
            if down_block.downsample_after:
                skip_connections.append(x)
                x = self.maxpool(x)

        x = self.middle_conv_block1(x, time_embedding)
        x = self.middle_attention_block1(x, time_embedding)
        x = self.middle_conv_block2(x, time_embedding) 
        upsampled = True

        skip_idx = 0
        for up_block in self.upsample_blocks:
            if upsampled:
                x = torch.cat((x, skip_connections[-skip_idx - 1]), dim=1)
                upsampled = False
                skip_idx+=1
            x = up_block(x, time_embedding)
            if up_block.upsample:
                upsampled = True

        x = self.output_conv(x)

        return x


if __name__ == "__main__":
    
    model = UNet()

    t = torch.rand((32, 1))
    batch = torch.ones((32, 3, 64, 64))

    inference = model(t, batch)
    print(inference.shape)