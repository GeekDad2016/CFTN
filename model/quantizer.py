import torch
import torch.nn as nn
from einops import einsum


class Quantizer(nn.Module):
    def __init__(self,
                 config
                 ):
        super(Quantizer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config['codebook_size'], config['latent_dim'])
        
        self.register_buffer('ema_cluster_size', torch.zeros(config['codebook_size']))
        self.ema_w = nn.Parameter(torch.Tensor(config['codebook_size'], config['latent_dim']))
        self.ema_w.data.normal_()
        
        self.decay = config.get('decay', 0.99)
        self.epsilon = config.get('epsilon', 1e-5)
        print(f"Type of self.epsilon: {type(self.epsilon)}, value: {self.epsilon}")

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x_flat = x.reshape(-1, C)
        
        dist = torch.cdist(x_flat, self.embedding.weight)
        min_encoding_indices = torch.argmin(dist, dim=1)
        min_encodings = torch.nn.functional.one_hot(min_encoding_indices, self.config['codebook_size']).type(x_flat.dtype)
        
        quant_out = torch.matmul(min_encodings, self.embedding.weight).view(x.shape)
        
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(min_encodings, 0)
            
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size.data = (self.ema_cluster_size.data + self.epsilon) \
                                         / (n + self.config['codebook_size'] * self.epsilon) * n
            
            dw = torch.matmul(min_encodings.t(), x_flat)
            self.ema_w.data = self.ema_w.data * self.decay + (1 - self.decay) * dw
            
            self.embedding.weight.data.copy_(self.ema_w.data / self.ema_cluster_size.data.unsqueeze(1))

        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)

        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
        }

        quant_out = x + (quant_out - x).detach()
        quant_out = quant_out.permute(0, 3, 1, 2).contiguous()
        
        min_encoding_indices = min_encoding_indices.reshape(x.shape[:-1])

        return quant_out, quantize_losses, min_encoding_indices
    
    def quantize_indices(self, indices):
        return einsum(indices, self.embedding.weight, 'b n h w, n d -> b d h w')


def get_quantizer(config):
    quantizer = Quantizer(
        config=config['model_params']
    )
    return quantizer

