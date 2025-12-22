import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.naruto_dataset import NarutoDataset
from model.vqvae_v2 import VQVAEv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def verify(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Load VQ-VAE
    vqvae_params = {k: v for k, v in config['model_params'].items() if k != 'image_size'}
    model = VQVAEv2(**vqvae_params).to(device)
    
    ckpt_path = os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name'])
    if not os.path.exists(ckpt_path):
        # Try relative to script
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ckpt_path)
        
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded VQ-VAE from {ckpt_path}")

    # 2. Load Dataset
    dataset = NarutoDataset(image_size=config['model_params']['image_size'])
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 3. Get a batch and reconstruct
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)
    
    with torch.no_grad():
        _, recons, _, _ = model(imgs)
        
    # 4. Save comparison
    comparison = torch.cat([imgs, recons], dim=0)
    comparison = (comparison * 0.5 + 0.5).clamp(0, 1)
    grid = make_grid(comparison, nrow=8)
    save_image(grid, "vqvae_verification.png")
    print("Verification image saved to vqvae_verification.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/vqvae_naruto.yaml')
    args = parser.parse_args()
    verify(args.config)
