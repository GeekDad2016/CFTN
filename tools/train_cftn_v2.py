import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb
import numpy as np
from torchvision.utils import make_grid, save_image
import argparse
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.naruto_dataset import NarutoDataset
from model.vqvae_v2 import VQVAEv2
from model.cftn_v2 import BiHemisphericBrain

# Silence tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_vqvae(config):
    vqvae_params = {k: v for k, v in config['model_params'].items() if k != 'image_size'}
    model = VQVAEv2(**vqvae_params).to(device)
    return model

def unnormalize(tensor):
    return tensor * 0.5 + 0.5

# ==========================================
# 1. Masking Functions
# ==========================================
def create_masked_image(indices, mask_token_id, min_ratio=0.1, max_ratio=1.0):
    """MaskGIT style masking for Images"""
    batch, seq_len = indices.shape
    device = indices.device
    
    # Random ratio per image
    ratios = torch.rand(batch, 1, device=device) * (max_ratio - min_ratio) + min_ratio
    noise = torch.rand(batch, seq_len, device=device)
    mask = noise < ratios
    
    masked_indices = indices.clone()
    masked_indices[mask] = mask_token_id
    
    return masked_indices, mask

def create_masked_text(text_ids, tokenizer, mask_prob=0.15):
    """BERT style masking for Text"""
    mask_token_id = tokenizer.mask_token_id
    batch, seq_len = text_ids.shape
    device = text_ids.device
    
    # Create mask (exclude padding and cls/sep if possible, simpler to just rand mask)
    # 15% probability of masking
    probs = torch.rand(batch, seq_len, device=device)
    mask = probs < mask_prob
    
    # Don't mask special tokens (0=PAD, 101=CLS, 102=SEP usually)
    special_tokens_mask = (text_ids == tokenizer.pad_token_id) | \
                          (text_ids == tokenizer.cls_token_id) | \
                          (text_ids == tokenizer.sep_token_id)
    mask = mask & ~special_tokens_mask
    
    masked_text = text_ids.clone()
    masked_text[mask] = mask_token_id
    
    return masked_text, mask

# ==========================================
# 2. Text Augmentation
# ==========================================
def apply_text_augmentations(captions, null_prob=0.10, word_drop_prob=0.1):
    new_captions = []
    for cap in captions:
        # Null Caption (CFG)
        if random.random() < null_prob:
            new_captions.append("") 
            continue
        
        # Word Dropout
        words = cap.split()
        if len(words) > 1 and word_drop_prob > 0:
            new_words = [w for w in words if random.random() > word_drop_prob]
            if not new_words: new_words = words
            new_captions.append(" ".join(new_words))
        else:
            new_captions.append(cap)
            
    return new_captions

@torch.no_grad()
def evaluate_brain(model, vq_model, tokenizer, device, config, tag, use_wandb):
    model.eval()
    
    eval_prompts = [
        "This is a close-up portrait of a male anime character with dark hair, a mustache, and intense, focused eyes. He wears a simple, dark-colored tunic with a laced neckline. The background is dark and shadowy, suggesting a cave or dimly lit environment.",
        "This is a close-up of a character, wearing a white mask with a red stripe on the forehead and a smiling expression. He has spiky, light brown hair and is dressed in a dark high-collared outfit. The background is a blurred green forest.",
        "A young ninja with vibrant red hair and striking blue eyes, wearing a forehead protector with a metal plate and a light-colored scarf. The background is a blurred green forest.",
        "A close-up of an anime character with spiky black hair, dark sunglasses, and a goatee, wearing a high-collared beige hoodie with a zipper down the front. The background is a blurred green forest."
    ]
    
    text_ids = tokenizer(eval_prompts, padding='max_length', truncation=True, 
                         max_length=config['cftn_v2_params']['text_block_size'], 
                         return_tensors="pt").input_ids.to(device)
    
    # 1-Shot Inference (Full Mask)
    mask_token_id = config['model_params']['num_embeddings']
    batch_size = len(eval_prompts)
    seq_len = config['cftn_v2_params']['block_size']
    
    full_mask = torch.full((batch_size, seq_len), mask_token_id, dtype=torch.long, device=device)
    
    # Predict
    _, vis_logits = model(img_indices=full_mask, text_ids=text_ids)
    pred_indices = torch.argmax(vis_logits, dim=-1)
    
    # Decode
    gen_imgs = vq_model.decode_indices(pred_indices)
    gen_imgs = unnormalize(gen_imgs).clamp(0,1)

    results_dir = os.path.join(config['train_params']['task_name'], "cftn_v2_gen_results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"gen_{tag}.png")
    save_image(gen_imgs, save_path, nrow=4)

    print(f"📊 Eval {tag} | Saved Grid: {save_path}")
    
    if use_wandb:
        caption_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(eval_prompts)])
        wandb.log({
            "eval/generations": wandb.Image(save_path, caption=caption_text)
        })
    
    model.train()

def train(args):
    # 1. Load Config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Setup
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = NarutoDataset(image_size=config['model_params']['image_size'])
    loader = DataLoader(dataset, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    
    vqvae = get_vqvae(config)
    vqvae_ckpt = os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name'])
    if os.path.exists(vqvae_ckpt):
        print(f"Loading VQ-VAE checkpoint from {vqvae_ckpt}")
        vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device, weights_only=True))
    vqvae.eval()
    for p in vqvae.parameters(): p.requires_grad = False

    # 3. Model Setup (Brain)
    img_size = config['model_params']['image_size']
    downsample = 2 ** config['model_params']['num_downsampling_layers']
    block_size = (img_size // downsample) ** 2
    
    # MASK ID = num_embeddings (last index)
    mask_token_id = config['model_params']['num_embeddings']
    
    config['cftn_v2_params'].update({
        'block_size': block_size,
        'vocab_size': config['model_params']['num_embeddings'] + 1, # +1 for Mask Token
        'text_vocab_size': tokenizer.vocab_size
    })
    
    model = BiHemisphericBrain(config['cftn_v2_params']).to(device)
    
    # Optimizer
    gate_params = [p for n, p in model.named_parameters() if 'gate' in n]
    other_params = [p for n, p in model.named_parameters() if 'gate' not in n]
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': config['cftn_v2_params']['lr_text']}, 
        {'params': gate_params, 'lr': config['cftn_v2_params']['lr_gate']}
    ])
    
    # Loss with Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Resume
    start_epoch = 0
    checkpoint_path = os.path.join(config['train_params']['task_name'], "best_cftn_v2_gen.pth")
    
    if os.path.exists(checkpoint_path):
        try:
            print(f"Loading Checkpoint from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch']
        except Exception as e:
            print(f"⚠️ Load failed (vocab mismatch?): {e}. Starting Fresh.")

    if args.use_wandb:
        wandb.init(project="vqvae-naruto-cftn-v2", config=config, name="Stage2_Track_Towers")

    print("🚀 STAGE 2: MaskGIT Training (Tracking Tower Communication)...")
    
    epochs = config['cftn_v2_params'].get('epochs_gen_only', 200)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        pbar = tqdm(loader)
        
        # Trackers
        ep_img_loss, ep_txt_loss, ep_img_acc, ep_txt_acc = [], [], [], []
        gate_stats = {} # To store mean gate values per batch
        
        for imgs, captions in pbar:
            imgs = imgs.to(device)
            
            # ---------------------------------------------
            # 1. PREPARE INPUTS
            # ---------------------------------------------
            
            # A. Text (Masked for bidirectional learning)
            aug_captions = apply_text_augmentations(list(captions))
            clean_text_ids = tokenizer(aug_captions, padding='max_length', truncation=True, 
                                       max_length=config['cftn_v2_params']['text_block_size'], 
                                       return_tensors="pt").input_ids.to(device)
            masked_text_ids, text_mask = create_masked_text(clean_text_ids, tokenizer)
            
            # B. Images (Masked for MaskGIT)
            with torch.no_grad(): 
                target_img_indices = vqvae.get_indices(imgs) 
            masked_img_indices, img_mask = create_masked_image(target_img_indices, mask_token_id)
            
            optimizer.zero_grad()
            
            # ---------------------------------------------
            # 2. FORWARD PASS
            # ---------------------------------------------
            # We pass MASKED images and MASKED text. 
            # The model attempts to reconstruct both.
            txt_logits, vis_logits = model(img_indices=masked_img_indices, text_ids=masked_text_ids)
            
            # ---------------------------------------------
            # 3. LOSS CALCULATION
            # ---------------------------------------------
            
            # A. Image Loss (Reconstruct masked image tokens)
            # Flatten to [Batch*Seq, Vocab]
            loss_img = criterion(
                vis_logits.reshape(-1, config['cftn_v2_params']['vocab_size']), 
                target_img_indices.reshape(-1)
            )
            
            # B. Text Loss (Reconstruct masked text tokens)
            # We only care about the masked tokens usually, but full seq loss helps stability
            loss_txt = criterion(
                txt_logits.reshape(-1, tokenizer.vocab_size),
                clean_text_ids.reshape(-1)
            )
            
            # Total Loss (Balance the two towers)
            # Weight text less (0.5) since our main goal is image generation
            loss = loss_img + (0.5 * loss_txt)
            
            loss.backward()
            optimizer.step()
            
            # ---------------------------------------------
            # 4. STATS & LOGGING
            # ---------------------------------------------
            
            # Accuracies
            img_preds = torch.argmax(vis_logits, dim=-1)
            img_acc = (img_preds == target_img_indices).float().mean().item() * 100
            
            txt_preds = torch.argmax(txt_logits, dim=-1)
            # Calculate accuracy only on non-padding tokens
            valid_txt = clean_text_ids != tokenizer.pad_token_id
            txt_acc = (txt_preds[valid_txt] == clean_text_ids[valid_txt]).float().mean().item() * 100
            
            ep_img_loss.append(loss_img.item())
            ep_txt_loss.append(loss_txt.item())
            ep_img_acc.append(img_acc)
            ep_txt_acc.append(txt_acc)
            
            # --- TRACK GATES (The "Talk" between Towers) ---
            # We iterate named_parameters to find gates and log their mean value
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "gate" in name and param.requires_grad:
                        # Store scalar value
                        val = param.mean().item()
                        if name not in gate_stats: gate_stats[name] = []
                        gate_stats[name].append(val)

            pbar.set_description(f"Ep {epoch} | ImgL: {np.mean(ep_img_loss):.3f} | TxtL: {np.mean(ep_txt_loss):.3f}")

        # End of Epoch Logging
        log_dict = {
            "loss/image": np.mean(ep_img_loss),
            "loss/text": np.mean(ep_txt_loss),
            "acc/image": np.mean(ep_img_acc),
            "acc/text": np.mean(ep_txt_acc),
            "epoch": epoch
        }
        
        # Add averaged gate values to log
        for name, vals in gate_stats.items():
            # Clean up name for wandb (e.g. layers.0.gate -> gates/layer_0)
            clean_name = f"gates/{name.replace('.', '_')}"
            log_dict[clean_name] = np.mean(vals)
            
        if args.use_wandb:
            wandb.log(log_dict)

        # Save Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        if epoch % 2 == 0:
            evaluate_brain(model, vqvae, tokenizer, device, config, f"track_{epoch}", args.use_wandb)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='config/vqvae_naruto.yaml')
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()
    train(args)