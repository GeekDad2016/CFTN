import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.cftn import CFTN
from model.vqvae_v2 import VQVAEv2
import yaml
from torchvision.utils import save_image, make_grid
import math
import os
import argparse
import wandb

# Silence tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_vqvae(config):
    # Filter out image_size which is used for dataset but not VQVAE init
    vqvae_params = {k: v for k, v in config['model_params'].items() if k != 'image_size'}
    model = VQVAEv2(**vqvae_params).to(device)
    return model

@torch.no_grad()
def generate(prompt, steps=12, num_samples=4):
    with open('config/vqvae_naruto.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load Models
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    config['transformer_params'].update({
        'block_size': 1024, 
        'vocab_size': config['model_params']['num_embeddings'],
        'text_vocab_size': tokenizer.vocab_size, 
        'text_block_size': 128
    })
    
    model = CFTN(config['transformer_params']).to(device)
    model_path = os.path.join(config['train_params']['task_name'], "best_cftn.pth")
    if os.path.exists(model_path):
        print(f"Loading CFTN checkpoint from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    vqvae = get_vqvae(config)
    vqvae_ckpt = os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name'])
    if os.path.exists(vqvae_ckpt):
        print(f"Loading VQ-VAE checkpoint from {vqvae_ckpt}")
        vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device, weights_only=True))
    vqvae.eval()
    
    wandb.init(project="vqvae-naruto-cftn-gen", config={"prompt": prompt, "steps": steps, "num_samples": num_samples})
    
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
        
        # Keep the most confident tokens, mask the rest
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
    
    grid = make_grid(output, nrow=int(math.sqrt(num_samples)))
    
    results_dir = os.path.join(config['train_params']['task_name'], "cftn_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    save_path = os.path.join(results_dir, "generation_final.png")
    save_image(grid, save_path)
    wandb.log({"generated_images": wandb.Image(save_path)})
    print(f"Image saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A high quality digital art of Naruto")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--num_samples", type=int, default=4)
    args = parser.parse_args()
    generate(args.prompt, args.steps, args.num_samples)