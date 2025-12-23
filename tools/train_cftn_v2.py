import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
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
    # Filter out image_size which is used for dataset but not VQVAE init
    vqvae_params = {k: v for k, v in config['model_params'].items() if k != 'image_size'}
    model = VQVAEv2(**vqvae_params).to(device)
    return model

def unnormalize(tensor):
    return tensor * 0.5 + 0.5

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2, reduction='none').mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-8) 
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.mean().item()

@torch.no_grad()
def evaluate_brain(model, vq_model, tokenizer, device, val_loader, config, tag, use_wandb):
    model.eval()
    
    total_psnr = 0
    # 1. PSNR Calculation
    for imgs, captions in val_loader:
        imgs = imgs.to(device)
        text_ids = tokenizer(captions, padding='max_length', truncation=True, 
                             max_length=config['cftn_v2_params']['text_block_size'], 
                             return_tensors="pt").input_ids.to(device)
        
        # Predict tokens from text
        _, vis_logits = model(img_indices=None, text_ids=text_ids)
        pred_indices = torch.argmax(vis_logits, dim=-1) 
        
        gen_imgs = vq_model.decode_indices(pred_indices)
        gen_imgs = unnormalize(gen_imgs).clamp(0,1)
        
        gt_indices = vq_model.get_indices(imgs)
        gt_rec_imgs = vq_model.decode_indices(gt_indices)
        gt_rec_imgs = unnormalize(gt_rec_imgs).clamp(0,1)

        total_psnr += calculate_psnr(gen_imgs, gt_rec_imgs)
    
    avg_psnr = total_psnr / len(val_loader)
    
    # 2. Sample Generation
    test_prompt = "Naruto standing in the forest, high quality digital art"
    tokens = tokenizer(test_prompt, return_tensors="pt", padding='max_length', 
                       max_length=config['cftn_v2_params']['text_block_size']).input_ids.to(device)
    
    _, vis_logits = model(img_indices=None, text_ids=tokens)
    pred_indices = torch.argmax(vis_logits, dim=-1)
    custom_img = vq_model.decode_indices(pred_indices)
    custom_img = unnormalize(custom_img).clamp(0,1)

    # 3. Validation Grid
    val_iter = iter(val_loader)
    v_imgs, v_captions = next(val_iter)
    v_text_ids = tokenizer(v_captions[:8], padding='max_length', truncation=True, 
                           max_length=config['cftn_v2_params']['text_block_size'], 
                           return_tensors="pt").input_ids.to(device)
    
    _, v_logits = model(img_indices=None, text_ids=v_text_ids)
    v_pred = torch.argmax(v_logits, dim=-1)
    v_gen = vq_model.decode_indices(v_pred)
    v_gen = unnormalize(v_gen).clamp(0,1)

    vis_grid = torch.cat([custom_img, v_gen[:7]], dim=0)
    
    results_dir = os.path.join(config['train_params']['task_name'], "cftn_v2_results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"eval_{tag}.png")
    save_image(vis_grid, save_path, nrow=4)

    print(f"📊 Eval {tag} | PSNR: {avg_psnr:.2f} | Saved: {save_path}")
    if use_wandb:
        wandb.log({
            "brain/eval_psnr": avg_psnr,
            "brain/generation": wandb.Image(save_path, caption=f"Tag: {tag}")
        })
    
    model.train()

def train(args):
    # 1. Load Config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Setup Tokenizer & Dataset
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
    for p in vqvae.parameters(): p.requires_grad = False

    # 4. Setup Brain Model
    config['cftn_v2_params'].update({
        'block_size': config['cftn_params']['block_size'],
        'vocab_size': config['model_params']['num_embeddings'],
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
    
    criterion = nn.CrossEntropyLoss()
    
    # Validation Subset
    val_indices = list(range(min(16, len(dataset))))
    val_ds = Subset(dataset, val_indices)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Resume Logic
    start_epoch = 0
    stage = '2A_Bidirectional'
    checkpoint_path = os.path.join(config['train_params']['task_name'], "best_cftn_v2.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading CFTN v2 checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        stage = ckpt.get('stage', '2A_Bidirectional')

    if args.use_wandb:
        wandb.init(project="vqvae-naruto-cftn-v2", config=config)

    # --- STAGE 2A: Bidirectional Training ---
    if stage == '2A_Bidirectional':
        print("🚀 STAGE 2A: Bidirectional Training (Gen + Captioning)...")
        for epoch in range(start_epoch, config['cftn_v2_params']['epochs_bidirectional']):
            model.train()
            pbar = tqdm(loader)
            loss_gen, loss_cap, acc_gen = [], [], []
            
            for imgs, captions in pbar:
                imgs = imgs.to(device)
                text_ids = tokenizer(captions, padding='max_length', truncation=True, 
                                     max_length=config['cftn_v2_params']['text_block_size'], 
                                     return_tensors="pt").input_ids.to(device)
                
                with torch.no_grad(): 
                    img_indices = vqvae.get_indices(imgs) 
                
                optimizer.zero_grad()
                
                # 50% Generation, 50% Captioning
                if random.random() < 0.5:
                    _, vis_logits = model(img_indices=None, text_ids=text_ids)
                    loss = criterion(vis_logits.reshape(-1, config['model_params']['num_embeddings']), img_indices.reshape(-1))
                    loss_gen.append(loss.item())
                    
                    preds = torch.argmax(vis_logits, dim=-1)
                    acc = (preds == img_indices).float().mean().item() * 100
                    acc_gen.append(acc)
                else:
                    # Captioning: predict next text token
                    txt_logits, _ = model(img_indices=img_indices, text_ids=text_ids[:, :-1])
                    loss = criterion(txt_logits.reshape(-1, tokenizer.vocab_size), text_ids[:, 1:].reshape(-1))
                    loss_cap.append(loss.item())

                loss.backward()
                optimizer.step()
                
                pbar.set_description(f"Ep {epoch} | GenL: {np.mean(loss_gen) if loss_gen else 0:.3f} | CapL: {np.mean(loss_cap) if loss_cap else 0:.3f} | Acc: {np.mean(acc_gen) if acc_gen else 0:.1f}%")

            if args.use_wandb:
                wandb.log({
                    "brain/gen_loss": np.mean(loss_gen) if loss_gen else 0, 
                    "brain/cap_loss": np.mean(loss_cap) if loss_cap else 0, 
                    "brain/vis_accuracy": np.mean(acc_gen) if acc_gen else 0,
                    "epoch": epoch,
                    "stage": "2A_Bidirectional"
                })
            
            if epoch % config['cftn_v2_params']['eval_every'] == 0:
                evaluate_brain(model, vqvae, tokenizer, device, val_loader, config, f"2A_{epoch}", args.use_wandb)
                
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stage': '2A_Bidirectional'
            }, checkpoint_path)
            
        stage = '2B_GenOnly'
        start_epoch = 0

    # --- STAGE 2B: Generation Fine-Tuning ---
    if stage == '2B_GenOnly':
        print("🚀 STAGE 2B: Fine-Tuning Generation Only...")
        for epoch in range(start_epoch, config['cftn_v2_params']['epochs_gen_only']):
            model.train()
            pbar = tqdm(loader)
            epoch_loss, epoch_acc = [], []
            
            for imgs, captions in pbar:
                imgs = imgs.to(device)
                text_ids = tokenizer(captions, padding='max_length', truncation=True, 
                                     max_length=config['cftn_v2_params']['text_block_size'], 
                                     return_tensors="pt").input_ids.to(device)
                
                with torch.no_grad(): 
                    img_indices = vqvae.get_indices(imgs) 
                
                optimizer.zero_grad()
                _, vis_logits = model(img_indices=None, text_ids=text_ids)
                loss = criterion(vis_logits.reshape(-1, config['model_params']['num_embeddings']), img_indices.reshape(-1))
                
                loss.backward()
                optimizer.step()
                
                preds = torch.argmax(vis_logits, dim=-1)
                acc = (preds == img_indices).float().mean().item() * 100
                epoch_acc.append(acc)
                epoch_loss.append(loss.item())
                
                pbar.set_description(f"Ep {epoch} | Loss: {np.mean(epoch_loss):.4f} | Acc: {np.mean(epoch_acc):.1f}%")

            if args.use_wandb:
                wandb.log({
                    "brain/img_loss": np.mean(epoch_loss),
                    "brain/vis_accuracy": np.mean(epoch_acc),
                    "epoch": epoch,
                    "stage": "2B_GenOnly"
                })

            if epoch % config['cftn_v2_params']['eval_every'] == 0:
                evaluate_brain(model, vqvae, tokenizer, device, val_loader, config, f"2B_{epoch}", args.use_wandb)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stage': '2B_GenOnly'
            }, checkpoint_path)
            
            if np.mean(epoch_loss) < config['cftn_v2_params']['target_img_loss']:
                print(f"✅ Target Loss {config['cftn_v2_params']['target_img_loss']} reached!")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='config/vqvae_naruto.yaml')
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()
    train(args)
