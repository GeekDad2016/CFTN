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
import argparse

# Silence tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_vqvae(config):
    # Filter out image_size which is used for dataset but not VQVAE init
    vqvae_params = {k: v for k, v in config['model_params'].items() if k != 'image_size'}
    model = VQVAEv2(**vqvae_params).to(device)
    return model

@torch.no_grad()
def generate_samples(model, vqvae, tokenizer, prompt, num_samples=16, steps=12, temp=1.0):
    model.eval()
    vqvae.eval()
    
    # 1. Prepare Inputs
    text_tokens = tokenizer([prompt] * num_samples, padding='max_length', truncation=True, max_length=128, return_tensors="pt").input_ids.to(device)
    
    # Start with all masked
    cur_ids = torch.full((num_samples, 1024), model.mask_token_id).long().to(device)
    cur_mask = torch.ones((num_samples, 1024), dtype=torch.bool, device=device)
    
    # 2. Iterative Parallel Decoding (MaskGIT logic)
    for i in range(steps):
        # Calculate how many tokens to mask (Cosine Schedule)
        # ratio is how many tokens should be MASKED at the end of this step
        ratio = 1.0 - math.cos(((i + 1) / steps) * (math.pi / 2))
        num_to_mask = max(0, int((1.0 - ratio) * 1024))
        
        logits = model(cur_ids, text_tokens)
        probs = F.softmax(logits / max(temp, 1e-6), dim=-1)
        
        # Sample predictions using Gumbel-max trick
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + 1e-10) + 1e-10)
        predictions = torch.argmax(logits / max(temp, 1e-6) + gumbel_noise, dim=-1)
        
        # Confidence is the probability of the predicted token
        confidences = torch.gather(probs, -1, predictions.unsqueeze(-1)).squeeze(-1)
        
        # If a token was already unmasked, we give it high confidence to keep it
        confidences[~cur_mask] = 2.0 
        
        # Identify which tokens to mask for the next iteration
        # We want to keep the (1024 - num_to_mask) most confident tokens
        new_mask = torch.ones_like(cur_mask)
        for b in range(num_samples):
            if num_to_mask < 1024:
                _, topk_indices = torch.topk(confidences[b], k=1024 - num_to_mask)
                new_mask[b, topk_indices] = False
        
        cur_mask = new_mask
        cur_ids = predictions.clone()
        cur_ids[cur_mask] = model.mask_token_id

    # 3. Final Safety Check - Fill all remaining masks
    if (cur_ids == model.mask_token_id).any():
        logits = model(cur_ids, text_tokens)
        predictions = torch.argmax(logits, dim=-1)
        mask = (cur_ids == model.mask_token_id)
        cur_ids[mask] = predictions[mask]

    # 4. Decode
    indices = cur_ids.view(num_samples, 32, 32)
    indices = torch.clamp(indices, 0, vqvae.vq.num_embeddings - 1)
    
    embeddings = vqvae.vq.embedding(indices).permute(0, 3, 1, 2)
    output = vqvae.decoder(embeddings)
    output = (output * 0.5 + 0.5).clamp(0, 1)
    
    model.train()
    return output

def train(args):
    # 1. Load Config
    with open(args.config_path, 'r') as f:
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
    # Automatically calculate block_size and vocab_size from VQVAE config
    config['cftn_params'].update({
        'block_size': 1024, # 32x32
        'vocab_size': config['model_params']['num_embeddings'],
        'text_vocab_size': tokenizer.vocab_size
    })
    
    model = CFTN(config['cftn_params']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['cftn_params']['lr'])
    
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
            model.load_state_dict(checkpoint)

    wandb.init(project="vqvae-naruto-cftn", config=config)

    test_prompt = "Naruto standing in the forest, high quality digital art"
    
    results_dir = os.path.join(config['train_params']['task_name'], "cftn_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    global_step = 0
    for epoch in range(start_epoch, config['cftn_params']['epochs']):
        pbar = tqdm(loader)
        epoch_losses = []
        epoch_accs = []
        
        for imgs, captions in pbar:
            imgs = imgs.to(device)
            
            # A. Get VQ-VAE Indices
            with torch.no_grad():
                _, _, _, indices = vqvae(imgs)
                indices = indices.view(imgs.size(0), -1) # [B, 1024]
            
            # B. Prepare Text
            text_tokens = tokenizer(captions, padding='max_length', truncation=True, max_length=config['cftn_params']['text_block_size'], return_tensors="pt").input_ids.to(device)
            
            # C. Masking Logic
            # MaskGIT style: cosine distribution for masking ratio
            # favors higher masking ratios
            u = np.random.uniform(0, 1)
            r = math.cos(u * math.pi / 2)
            
            mask = torch.bernoulli(torch.full(indices.shape, r, device=device))
            
            # Ensure at least one token is masked to avoid zero loss
            if mask.sum() == 0:
                mask[0, 0] = 1
            
            masked_indices = indices.clone()
            masked_indices[mask == 1] = model.mask_token_id
            
            # D. Forward and Loss
            logits = model(masked_indices, text_tokens)
            
            # Cross entropy only on masked tokens
            loss_full = F.cross_entropy(logits.view(-1, logits.size(-1)), indices.view(-1), reduction='none')
            loss = (loss_full * mask.view(-1)).sum() / (mask.sum() + 1e-8)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # E. Metrics Calculation
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                # Accuracy only on masked tokens
                correct = (preds == indices) & (mask == 1)
                accuracy = correct.sum().float() / (mask.sum() + 1e-8)
                
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
            
            # F. Step Logging
            if global_step % 10 == 0:
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/step_accuracy": accuracy.item(),
                    "train/mask_ratio": r,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "global_step": global_step
                })
            
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f} | Acc: {accuracy.item():.4f}")
            global_step += 1
            
        mean_loss = np.mean(epoch_losses)
        mean_acc = np.mean(epoch_accs)
        wandb.log({
            "train/epoch_loss": mean_loss,
            "train/epoch_accuracy": mean_acc,
            "epoch": epoch
        })
        
        if epoch % 10 == 0:
            sample = generate_samples(model, vqvae, tokenizer, test_prompt, num_samples=16, steps=config['cftn_params']['steps'], temp=config['cftn_params']['temperature'])
            grid = make_grid(sample, nrow=4)
            save_path = os.path.join(results_dir, f"epoch_{epoch}.png")
            save_image(grid, save_path)
            wandb.log({"val/cftn_sample": wandb.Image(save_path)})

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='config/vqvae_naruto.yaml')
    args = parser.parse_args()
    train(args)
