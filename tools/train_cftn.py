import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from dataset.naruto_dataset import NarutoDataset
from model.vqvae_v2 import VQVAEv2
from model.cftn import CFTN
import wandb
import numpy as np
from torchvision.utils import make_grid, save_image
import math

# Silence tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_vqvae(config):
    # Filter out image_size which is used for dataset but not VQVAE init
    vqvae_params = {k: v for k, v in config['model_params'].items() if k != 'image_size'}
    model = VQVAEv2(**vqvae_params).to(device)
    return model

@torch.no_grad()
def generate_samples(model, vqvae, tokenizer, prompt, num_samples=16, steps=12):
    model.eval()
    vqvae.eval()
    
    # 1. Prepare Inputs
    text_tokens = tokenizer([prompt] * num_samples, padding='max_length', truncation=True, max_length=128, return_tensors="pt").input_ids.to(device)
    cur_ids = torch.full((num_samples, 1024), model.mask_token_id).long().to(device)
    
    # 2. Iterative Parallel Decoding
    for i in range(steps):
        # Calculate how many tokens to mask (Cosine Schedule)
        ratio = 1.0 - math.cos(((i + 1) / steps) * (math.pi / 2))
        num_to_keep = max(1, int(ratio * 1024))
        
        logits = model(cur_ids, text_tokens)
        probs = F.softmax(logits, dim=-1)
        
        confidences, predictions = torch.max(probs, dim=-1)
        
        # Keep the most confident tokens, mask the rest per sample in batch
        new_ids = torch.full((num_samples, 1024), model.mask_token_id).long().to(device)
        for b in range(num_samples):
            _, topk_indices = torch.topk(confidences[b], k=num_to_keep)
            new_ids[b, topk_indices] = predictions[b, topk_indices]
        cur_ids = new_ids

    # 3. Final Safety Check: Replace any remaining mask tokens with most likely predictions
    if (cur_ids == model.mask_token_id).any():
        logits = model(cur_ids, text_tokens)
        predictions = torch.argmax(logits, dim=-1)
        mask = (cur_ids == model.mask_token_id)
        cur_ids[mask] = predictions[mask]

    # 4. Decode
    indices = cur_ids.view(num_samples, 32, 32)
    # Ensure indices are within valid VQ-VAE range [0, num_embeddings-1]
    indices = torch.clamp(indices, 0, vqvae.vq.num_embeddings - 1)
    
    embeddings = vqvae.vq.embedding(indices).permute(0, 3, 1, 2)
    output = vqvae.decoder(embeddings)
    output = (output * 0.5 + 0.5).clamp(0, 1)
    
    model.train()
    return output

def train():
    # 1. Load Config
    with open('config/vqvae_naruto.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Setup Tokenizers & Dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = NarutoDataset(image_size=config['model_params']['image_size'])
    loader = DataLoader(dataset, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    
    # 3. Load VQ-VAE (Pre-trained)
    vqvae = get_vqvae(config)
    vqvae_ckpt = os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name'])
    if os.path.exists(vqvae_ckpt):
        print(f"Loading VQ-VAE checkpoint from {vqvae_ckpt}")
        vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device, weights_only=True))
    vqvae.eval()

    # 4. Setup CFTN
    config['transformer_params'].update({
        'block_size': 1024, 
        'vocab_size': config['model_params']['num_embeddings'],
        'text_vocab_size': tokenizer.vocab_size,
        'text_block_size': 128
    })
    model = CFTN(config['transformer_params']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['transformer_params']['lr'])
    
    # 5. Load Checkpoint if found (Resume Logic)
    start_epoch = 0
    checkpoint_path = os.path.join(config['train_params']['task_name'], "best_cftn.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading CFTN checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            # Compatibility for weights-only checkpoints
            model.load_state_dict(checkpoint)
            print("Warning: Loaded weights only, optimizer state and epoch count reset.")

    wandb.init(project="vqvae-naruto-cftn", config=config)

    test_prompt = "Naruto standing in the forest, high quality digital art"
    
    results_dir = os.path.join(config['train_params']['task_name'], "cftn_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for epoch in range(start_epoch, config['transformer_params']['epochs']):
        pbar = tqdm(loader)
        epoch_losses = []
        for imgs, captions in pbar:
            imgs = imgs.to(device)
            
            # A. Get Image Tokens from VQ-VAE
            with torch.no_grad():
                _, _, _, indices = vqvae(imgs)
                indices = indices.view(imgs.size(0), -1) # [B, 1024]
            
            # B. Get Text Tokens
            text_tokens = tokenizer(captions, padding='max_length', truncation=True, max_length=128, return_tensors="pt").input_ids.to(device)
            
            # C. Masking Logic (MaskGIT)
            r = np.random.uniform(0.1, 1.0)
            mask = torch.bernoulli(torch.full(indices.shape, r, device=device))
            
            masked_indices = indices.clone()
            masked_indices[mask == 1] = model.mask_token_id
            
            # D. Forward and Loss
            logits = model(masked_indices, text_tokens)
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), indices.view(-1), reduction='none')
            loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-8)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
            
        mean_loss = np.mean(epoch_losses)
        wandb.log({"train/cftn_loss": mean_loss, "epoch": epoch})
        
        # E. Validation generation
        if epoch % 10 == 0:
            sample = generate_samples(model, vqvae, tokenizer, test_prompt, num_samples=16)
            grid = make_grid(sample, nrow=4)
            save_path = os.path.join(results_dir, f"epoch_{epoch}.png")
            save_image(grid, save_path)
            wandb.log({"val/cftn_sample": wandb.Image(save_path)})

        # Save comprehensive checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

if __name__ == "__main__":
    train()