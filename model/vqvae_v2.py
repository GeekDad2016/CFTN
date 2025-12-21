import torch.nn as nn
import torch.nn.functional as F
from .quantizer_v2 import VectorQuantizerEMA

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                     for _ in range(self.num_residual_layers)])

    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_downsampling_layers, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()
        # Initial conv
        self.layers.append(nn.Conv2d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.ReLU(True))
        
        # Downsampling layers
        for _ in range(num_downsampling_layers - 1):
            self.layers.append(nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1))
            self.layers.append(nn.ReLU(True))
            num_hiddens = num_hiddens

        self.layers.append(nn.Conv2d(num_hiddens, embedding_dim, kernel_size=3, stride=1, padding=1))
        self.layers.append(ResidualStack(embedding_dim, embedding_dim, num_residual_layers, num_residual_hiddens))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_upsampling_layers, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(embedding_dim, num_hiddens, kernel_size=3, stride=1, padding=1))
        self.layers.append(ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens))

        # Upsampling layers
        for _ in range(num_upsampling_layers):
            self.layers.append(nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1))
            self.layers.append(nn.ReLU(True))
            num_hiddens = num_hiddens // 2
        
        self.layers.append(nn.ConvTranspose2d(num_hiddens, 3, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.Tanh())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VQVAEv2(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay):
        super(VQVAEv2, self).__init__()
        
        self.encoder = Encoder(3, num_hiddens, 2, num_residual_layers, num_residual_hiddens, embedding_dim)
        self.vq = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        self.decoder = Decoder(embedding_dim, num_hiddens, 2, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self.encoder(x)
        vq_loss, quantized, perplexity, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return vq_loss, x_recon, perplexity
