import torch
import torch.nn as nn
import os
import yaml
import argparse
import random
import numpy as np
from torch.optim import Adam
import pickle
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from model.vqvae_v2 import VQVAEv2
from model.transformer import VQVAETransformer
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VQVAETransformerDataset(Dataset):
    def __init__(self, config):
        encoding_path = os.path.join(config['train_params']['task_name'],
                                     config['train_params']['output_train_dir'],
                                     'encodings.pkl')
        assert os.path.exists(encoding_path), "No encodings found. Run infer_vqvae.py first."
        
        with open(encoding_path, 'rb') as f:
            self.encodings = pickle.load(f)
        
        self.num_embeddings = config['model_params']['num_embeddings']
        self.latent_shape = (self.encodings.shape[1], self.encodings.shape[2])
        self.block_size = self.encodings.shape[1] * self.encodings.shape[2]
        
        # Flatten encodings: [N, H, W] -> [N, H*W]
        self.data = self.encodings.reshape(self.encodings.shape[0], -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # We add a start token at the beginning
        # Input: [StartToken, code1, code2, ..., codeN-1]
        # Target: [code1, code2, ..., codeN]
        seq = self.data[idx]
        
        x = torch.cat([torch.tensor([self.num_embeddings]), seq[:-1]])
        y = seq.clone()
        
        return x.long(), y.long()

@torch.no_grad()
def generate_samples(model, vqvae_model, config, latent_shape, epoch):
    print(f"Generating Transformer samples at epoch {epoch}...")
    model.eval()
    vqvae_model.eval()
    
    num_embeddings = config['model_params']['num_embeddings']
    num_samples = 16
    block_size = latent_shape[0] * latent_shape[1]
    
    # Start with start tokens
    x = torch.ones((num_samples, 1), device=device).long() * num_embeddings
    
    for _ in range(block_size):
        logits = model(x[:, -config['transformer_params']['block_size']:])
        # Take logits of the last position
        logits = logits[:, -1, :] / 0.8 # Temperature
        probs = torch.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, ix), dim=1)
    
    # Remove start token and reshape
    generated_indices = x[:, 1:].reshape(num_samples, latent_shape[0], latent_shape[1])
    
    # Decode
    embeddings = vqvae_model.vq.embedding(generated_indices)
    embeddings = embeddings.permute(0, 3, 1, 2).contiguous()
    
    recon = vqvae_model.decoder(embeddings)
    recon = (recon * 0.5 + 0.5).clamp(0, 1)
    
    grid = make_grid(recon.cpu(), nrow=4)
    wandb.log({"transformer/samples": wandb.Image(grid), "epoch": epoch})
    model.train()

def train_transformer(args):
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    wandb.init(project="vqvae-naruto-transformer", config=config)
    
    dataset = VQVAETransformerDataset(config)
    loader = DataLoader(dataset, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    
    # Update config with calculated block size
    config['transformer_params']['block_size'] = dataset.block_size
    config['transformer_params']['vocab_size'] = config['model_params']['num_embeddings']
    
    model = VQVAETransformer(config['transformer_params']).to(device)
    
    # Load VQVAE for visualization
    vqvae_model = VQVAEv2(
        num_hiddens=config['model_params']['num_hiddens'],
        num_downsampling_layers=config['model_params']['num_downsampling_layers'],
        num_upsampling_layers=config['model_params']['num_upsampling_layers'],
        num_residual_layers=config['model_params']['num_residual_layers'],
        num_residual_hiddens=config['model_params']['num_residual_hiddens'],
        num_embeddings=config['model_params']['num_embeddings'],
        embedding_dim=config['model_params']['embedding_dim'],
        commitment_cost=config['model_params']['commitment_cost'],
        decay=config['model_params']['decay']
    ).to(device)
    
    vqvae_ckpt = os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name'])
    if os.path.exists(vqvae_ckpt):
        vqvae_model.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    vqvae_model.eval()

    optimizer = Adam(model.parameters(), lr=config['transformer_params']['lr'])
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    checkpoint_path = os.path.join(config['train_params']['task_name'], 'best_transformer.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading Transformer checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    model.train()
    for epoch in range(start_epoch, config['transformer_params']['epochs']):
        losses = []
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            # Flatten for CrossEntropy
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        mean_loss = np.mean(losses)
        print(f'Epoch {epoch} : {mean_loss}')
        wandb.log({"transformer/loss": mean_loss, "epoch": epoch})
        
        if epoch % 10 == 0:
            generate_samples(model, vqvae_model, config, dataset.latent_shape, epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='config/vqvae_naruto.yaml')
    args = parser.parse_args()
    train_transformer(args)
