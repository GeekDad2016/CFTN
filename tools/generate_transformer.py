import argparse
import yaml
import os
import torch
import pickle
import numpy as np
import random
from tqdm import tqdm
import torchvision
from model.vqvae_v2 import VQVAEv2
from model.transformer import VQVAETransformer
from torchvision.utils import make_grid
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def generate(args):
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    wandb.init(project="vqvae-naruto-transformer-gen", config=config)

    ######## Set the desired seed value #######
    seed = args.seed if args.seed is not None else config['train_params'].get('seed', 111)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #######################################
    
    # Calculate sizes
    encodings_path = os.path.join(config['train_params']['task_name'],
                                  config['train_params']['output_train_dir'],
                                  'encodings.pkl')
    # Load using torch.load with map_location='cpu' to handle GPU tensors in pickle
    encodings = torch.load(encodings_path, map_location='cpu')
    
    latent_shape = (encodings.shape[1], encodings.shape[2])
    block_size = latent_shape[0] * latent_shape[1]
    config['transformer_params']['block_size'] = block_size
    config['transformer_params']['vocab_size'] = config['model_params']['num_embeddings']
    
    # Load Models
    model = VQVAETransformer(config['transformer_params']).to(device)
    checkpoint_path = os.path.join(config['train_params']['task_name'], 'best_transformer.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
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
    vqvae_model.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    vqvae_model.eval()
    
    print(f"Generating {args.num_samples} samples with temperature {args.temp}...")
    
    # Autoregressive sampling
    x = torch.ones((args.num_samples, 1), device=device).long() * config['model_params']['num_embeddings']
    for _ in tqdm(range(block_size)):
        logits = model(x[:, -block_size:])
        logits = logits[:, -1, :] / args.temp
        probs = torch.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, ix), dim=1)
    
    generated_indices = x[:, 1:].reshape(args.num_samples, latent_shape[0], latent_shape[1])
    
    # Decode to pixels
    embeddings = vqvae_model.vq.embedding(generated_indices)
    embeddings = embeddings.permute(0, 3, 1, 2).contiguous()
    output = vqvae_model.decoder(embeddings)
    output = (output * 0.5 + 0.5).clamp(0, 1)
    
    grid = make_grid(output.cpu(), nrow=int(math.sqrt(args.num_samples)) if args.num_samples > 1 else 1)
    wandb.log({"generated/final": wandb.Image(grid)})
    
    save_path = os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir'], 'transformer_gen.png')
    torchvision.utils.save_image(grid, save_path)
    print(f"Results saved to {save_path}")

if __name__ == '__main__':
    import math
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='config/vqvae_naruto.yaml')
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--temp', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    generate(args)
