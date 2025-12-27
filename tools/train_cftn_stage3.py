import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb
import numpy as np
from torchvision.utils import make_grid, save_image
import argparse
import lpips # Perceptual Loss

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.naruto_dataset import NarutoDataset
from model.vqvae_v2 import VQVAEv2
from model.cftn_v2 import BiHemisphericBrain

# Silence tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. The Critic (Reward Function)
# ==========================================
class RLCritic(nn.Module):
    def __init__(self, sharpness_weight=0.05):
        super().__init__()
        # LPIPS: Perceptual Distance (VGG based). 0.0 = Identical, 1.0 = Different
        print("   ⚖️ Loading LPIPS Critic (VGG)...")
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
        self.sharpness_weight = sharpness_weight

    def laplacian_variance(self, images):
        """Calculates edge sharpness. Higher variance = sharper edges."""
        # Convert to grayscale for edge detection
        gray = 0.299 * images[:, 0, :, :] + 0.587 * images[:, 1, :, :] + 0.114 * images[:, 2, :, :]
        gray = gray.unsqueeze(1)
        
        # Laplacian Kernel
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                              dtype=torch.float32, device=images.device).unsqueeze(0).unsqueeze(0)
        
        edge_maps = F.conv2d(gray, kernel, padding=1)
        return edge_maps.var(dim=[1, 2, 3]) # Variance per image

    @torch.no_grad()
    def calculate_reward(self, generated_img, target_img):
        """
        Reward = -LPIPS + (Sharpness * lambda)
        We want to Minimize LPIPS (distance) and Maximize Sharpness.
        """
        # 1. Perceptual Distance (Lower is better, so we negate it)
        # generated_img and target_img should be in range [-1, 1]
        p_loss = self.lpips_fn(generated_img, target_img).squeeze() # Shape: [Batch]
        
        # 2. Edge Sharpness (Higher is better)
        # FIX: Clamp to prevent noise explosion (Reward Hacking)
        raw_sharpness = self.laplacian_variance(generated_img)
        sharpness = torch.clamp(raw_sharpness, max=150.0) 
        
        # Total Reward
        # We negate p_loss because RL maximizes reward.
        reward = -p_loss + (self.sharpness_weight * sharpness)
        return reward, p_loss.mean().item(), raw_sharpness.mean().item()

# ==========================================
# 2. Utilities
# ==========================================
def unnormalize(tensor):
    return tensor * 0.5 + 0.5

def get_vqvae(config):
    vqvae_params = {k: v for k, v in config['model_params'].items() if k != 'image_size'}
    model = VQVAEv2(**vqvae_params).to(device)
    return model

# ==========================================
# 3. RL Training Loop
# ==========================================
def train_rl(args):
    # --- Load Config ---
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # --- Setup Data & Models ---
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    full_dataset = NarutoDataset(image_size=config['model_params']['image_size'])
    loader = DataLoader(full_dataset, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=2)
    
    # 1. Load VQ-VAE (Frozen)
    vqvae = get_vqvae(config)
    vqvae_ckpt = os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name'])
    if os.path.exists(vqvae_ckpt):
        vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device, weights_only=True))
    vqvae.eval()
    for p in vqvae.parameters(): p.requires_grad = False

    # 2. Load Brain (Pre-trained Phase 2)
    # We construct the model config
    img_size = config['model_params']['image_size']
    downsample = 2 ** config['model_params']['num_downsampling_layers']
    block_size = (img_size // downsample) ** 2
    
    config['cftn_v2_params'].update({
        'block_size': block_size,
        'vocab_size': config['model_params']['num_embeddings'],
        'text_vocab_size': tokenizer.vocab_size
    })
    
    model = BiHemisphericBrain(config['cftn_v2_params']).to(device)
    
    # --- Load Weights ---
    # First, try to load the Brain from Stage 2 (The "Blob" Model)
    brain_ckpt = os.path.join(config['train_params']['task_name'], "best_cftn_v2.pth")
    if os.path.exists(brain_ckpt):
        print(f"🧠 Loading Stage 2 Brain from {brain_ckpt}")
        ckpt = torch.load(brain_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("⚠️ No Stage 2 checkpoint found! You should train Phase 2 first.")
        return

    # --- FREEZE LEFT HEMISPHERE (Semantic Preservation) ---
    # We assume the Left Tower (Text) understands the prompt perfectly. 
    # We only want to train the Right Tower (Vision) to be sharper.
    for name, param in model.named_parameters():
        if "layers_L" in name or "text_embed" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True # Train Vision + Gates

    # Optimizer (Smaller LR for RL)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-5) # Very low LR for stability
    
    # The Critic (Low weight to prevent hacking)
    critic = RLCritic(sharpness_weight=0.02)

    # --- RESUME LOGIC (Fixing the restart issue) ---
    rl_ckpt_path = os.path.join(config['train_params']['task_name'], "best_cftn_rl.pth")
    start_epoch = 0
    if os.path.exists(rl_ckpt_path):
        print(f"🔄 Found previous RL Checkpoint: {rl_ckpt_path}")
        try:
            rl_ckpt = torch.load(rl_ckpt_path, map_location=device)
            model.load_state_dict(rl_ckpt['model_state_dict'])
            optimizer.load_state_dict(rl_ckpt['optimizer_state_dict'])
            start_epoch = rl_ckpt['epoch'] + 1
            print(f"   ⏩ Resuming from Epoch {start_epoch}")
        except Exception as e:
            print(f"   ⚠️ Error loading RL checkpoint (starting fresh): {e}")

    if args.use_wandb:
        wandb.init(project="vqvae-naruto-rl-v3", config=config, name="Step3_RL_Refiner", resume="allow")

    print("\n🚀 STAGE 3: RL Fine-Tuning (The Refiner)...")
    print(f"   Model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")

    # --- Training Loop ---
    epochs = 500 
    
    for epoch in range(start_epoch, epochs):
        model.train()
        pbar = tqdm(loader)
        
        epoch_rewards = []
        epoch_lpips = []
        epoch_sharp = []
        
        for imgs, captions in pbar:
            imgs = imgs.to(device) # Target Real Images
            
            # Prepare Text
            text_ids = tokenizer(captions, padding='max_length', truncation=True, 
                                 max_length=config['cftn_v2_params']['text_block_size'], 
                                 return_tensors="pt").input_ids.to(device)
            
            optimizer.zero_grad()
            
            # ============================================
            # A. The Baseline (Greedy / "The Blob")
            # ============================================
            with torch.no_grad():
                # 1. Forward Pass (No Gradients)
                _, logits_baseline = model(img_indices=None, text_ids=text_ids)
                
                # 2. Argmax (Greedy Selection)
                indices_baseline = torch.argmax(logits_baseline, dim=-1)
                
                # 3. Decode & Score
                imgs_baseline = vqvae.decode_indices(indices_baseline).clamp(-1, 1)
                rewards_baseline, _, _ = critic.calculate_reward(imgs_baseline, imgs)

            # ============================================
            # B. The Policy (Stochastic / "The Explorer")
            # ============================================
            # 1. Forward Pass (With Gradients)
            _, logits_policy = model(img_indices=None, text_ids=text_ids)
            
            # 2. Sample from Distribution
            probs = F.softmax(logits_policy, dim=-1)
            dist = Categorical(probs)
            
            # Sample actions (indices)
            indices_sampled = dist.sample() # Shape: [Batch, 256]
            
            # 3. Decode & Score
            # We must detach vqvae here just in case, though it's set to eval
            imgs_sampled = vqvae.decode_indices(indices_sampled).clamp(-1, 1)
            rewards_sampled, lpips_score, sharp_score = critic.calculate_reward(imgs_sampled, imgs)
            
            # ============================================
            # C. The Optimization (Policy Gradient)
            # ============================================
            
            # Calculate Advantage
            advantage = rewards_sampled - rewards_baseline
            
            # Log Probabilities of the actions we took
            log_probs = dist.log_prob(indices_sampled) # Shape: [Batch, 256]
            
            # KL Divergence Penalty (Anti-Drift)
            # Penalizes the model if it diverges too far from the "safe" baseline logic.
            # This keeps the image coherent (like Naruto) rather than exploding into noise.
            kl_div = F.kl_div(
                F.log_softmax(logits_policy, dim=-1), 
                F.softmax(logits_baseline, dim=-1), 
                reduction='batchmean'
            )

            # Combined Loss
            # 1. Policy Gradient: -(log_probs * advantage) -> Maximize Reward
            # 2. KL Penalty: + (0.05 * kl_div) -> Minimize Drift/Noise
            kl_weight = 0.05 
            pg_loss = -(log_probs.sum(dim=1) * advantage).mean()
            loss = pg_loss + (kl_weight * kl_div)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0) # Crucial for RL stability
            optimizer.step()
            
            # Logging
            epoch_rewards.append(rewards_sampled.mean().item())
            epoch_lpips.append(lpips_score)
            epoch_sharp.append(sharp_score)
            
            pbar.set_description(f"Ep {epoch} | LPIPS: {np.mean(epoch_lpips):.3f} | Sharp: {np.mean(epoch_sharp):.1f} | Adv: {advantage.mean().item():.3f}")

        # --- End of Epoch Eval ---
        if args.use_wandb:
            wandb.log({
                "rl/reward": np.mean(epoch_rewards),
                "rl/lpips": np.mean(epoch_lpips),
                "rl/sharpness": np.mean(epoch_sharp),
                "rl/advantage_mean": advantage.mean().item()
            })
            
        # Save Checkpoint
        save_path = os.path.join(config['train_params']['task_name'], "best_cftn_rl.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stage': '3_RL_Refiner'
        }, save_path)
        
        # Visual Check (Save Greedy vs Sampled comparison)
        if epoch % 5 == 0:
            with torch.no_grad():
                vis_grid = torch.cat([unnormalize(imgs_baseline[:4]), unnormalize(imgs_sampled[:4])], dim=0)
                save_file = os.path.join(config['train_params']['task_name'], f"rl_epoch_{epoch}.png")
                save_image(vis_grid, save_file, nrow=4)
                print(f"   📸 Saved RL comparison: {save_file}")
                if args.use_wandb:
                    wandb.log({
                        "eval/rl_image": wandb.Image(save_file, caption=f"Epoch: {epoch}")
                    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='config/vqvae_naruto.yaml')
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()
    train_rl(args)