import yaml
import argparse
import torch
import random
import os
import numpy as np
from tqdm import tqdm
from model.vqvae_v2 import VQVAEv2
from torch.utils.data.dataloader import DataLoader
from dataset.naruto_dataset import NarutoDataset
from torch.optim import Adam
from torchvision.utils import make_grid, save_image
import wandb
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataset(config):
    if config['train_params']['dataset'] == 'naruto':
        dataset = NarutoDataset(image_size=config['model_params']['image_size'])
    else:
        raise ValueError("Unknown dataset")
    return dataset

def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    wandb.init(project="vqvae-naruto-v2", config=config)
    
    print(config)
    print(f"Using device: {device}")
    
    seed = config['train_params']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    model = VQVAEv2(
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
    
    dataset = get_dataset(config)
    data_loader = DataLoader(dataset, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    
    val_images, _ = next(iter(DataLoader(dataset, batch_size=16, shuffle=True)))
    val_images = val_images.to(device)

    num_epochs = config['train_params']['epochs']
    optimizer = Adam(model.parameters(), lr=config['train_params']['lr'])
    
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])
    results_dir = os.path.join(config['train_params']['task_name'], 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if os.path.exists(os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name'])):
        print('Loading checkpoint')
        model.load_state_dict(torch.load(os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name']), map_location=device))
    
    for epoch_idx in range(num_epochs):
        model.train()
        for i, (imgs, _) in enumerate(tqdm(data_loader)):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            
            vq_loss, recon_img, perplexity = model(imgs)
            recon_loss = F.mse_loss(recon_img, imgs)
            total_loss = recon_loss + vq_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if i % 100 == 0:
                wandb.log({
                    "epoch": epoch_idx,
                    "step": i,
                    "train/total_loss": total_loss.item(), 
                    "train/recon_loss": recon_loss.item(),
                    "train/vq_loss": vq_loss.item(),
                    "train/perplexity": perplexity.item()
                })
        
        print(f"Epoch {epoch_idx} | Recon Loss: {recon_loss.item():.4f} | VQ Loss: {vq_loss.item():.4f} | Perplexity: {perplexity.item():.2f}")
        
        model.eval()
        with torch.no_grad():
            _, recon_sample, _, = model(val_images)
            disp_orig = val_images * 0.5 + 0.5
            disp_recon = recon_sample.clamp(0,1) * 0.5 + 0.5
            grid = make_grid(torch.cat([disp_orig, disp_recon]), nrow=8)
            save_path = f"{results_dir}/epoch_{epoch_idx}.png"
            save_image(grid, save_path)
            wandb.log({"eval/recon_grid": wandb.Image(save_path)})

        torch.save(model.state_dict(), os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path', default='config/vqvae_naruto.yaml', type=str)
    args = parser.parse_args()
    train(args)
