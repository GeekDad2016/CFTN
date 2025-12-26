import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
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

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2, reduction='none').mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-8) 
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.mean().item()

@torch.no_grad()
def evaluate_brain(model, vq_model, tokenizer, device, val_loader, config, tag, use_wandb):
    model.eval()
    
    # --- Part 1: PSNR Calculation (Image Metrics) ---
    total_psnr = 0
    # Loop over validation loader to calculate average PSNR
    for imgs, captions in val_loader:
        imgs = imgs.to(device)
        text_ids = tokenizer(captions, padding='max_length', truncation=True, 
                             max_length=config['cftn_v2_params']['text_block_size'], 
                             return_tensors="pt").input_ids.to(device)
        
        _, vis_logits = model(img_indices=None, text_ids=text_ids)
        pred_indices = torch.argmax(vis_logits, dim=-1) 
        gen_imgs = vq_model.decode_indices(pred_indices)
        
        gt_indices = vq_model.get_indices(imgs)
        gt_rec_imgs = vq_model.decode_indices(gt_indices)
        
        total_psnr += calculate_psnr(unnormalize(gen_imgs), unnormalize(gt_rec_imgs))
    
    avg_psnr = total_psnr / len(val_loader)
    
    # --- Part 2: Image Generation Comparison (Greedy vs Temp) ---
    test_prompt = "a person with blue hair in a military uniform standing in front of a building"
    tokens = tokenizer(test_prompt, return_tensors="pt", padding='max_length', 
                       max_length=config['cftn_v2_params']['text_block_size']).input_ids.to(device)
    
    _, vis_logits = model(img_indices=None, text_ids=tokens)
    B, S, V = vis_logits.shape
    flat_logits = vis_logits.view(-1, V) 
    
    # A. GREEDY
    pred_greedy = torch.argmax(vis_logits, dim=-1)
    img_greedy = unnormalize(vq_model.decode_indices(pred_greedy)).clamp(0,1)

    # B. TEMP 0.8
    probs_1 = torch.softmax(flat_logits / 0.8, dim=-1)
    pred_sampled_1 = torch.multinomial(probs_1, num_samples=1).view(B, S)
    img_sampled_1 = unnormalize(vq_model.decode_indices(pred_sampled_1)).clamp(0,1)

    # C. TEMP 1.0
    probs_2 = torch.softmax(flat_logits / 1.0, dim=-1)
    pred_sampled_2 = torch.multinomial(probs_2, num_samples=1).view(B, S)
    img_sampled_2 = unnormalize(vq_model.decode_indices(pred_sampled_2)).clamp(0,1)

    # --- Part 3: Validation Grid (Images) ---
    val_iter = iter(val_loader)
    v_imgs, v_captions = next(val_iter) # Get one batch
    
    # Prepare inputs for visualization
    v_text_ids = tokenizer(v_captions[:3], padding='max_length', truncation=True, 
                           max_length=config['cftn_v2_params']['text_block_size'], 
                           return_tensors="pt").input_ids.to(device)
    v_imgs_gpu = v_imgs[:3].to(device)
    
    # Generate images from validation captions
    _, v_logits = model(img_indices=None, text_ids=v_text_ids)
    v_gen_greedy = unnormalize(vq_model.decode_indices(torch.argmax(v_logits, dim=-1))).clamp(0,1)
    
    # Save Image Grid
    vis_grid = torch.cat([img_greedy, img_sampled_1, img_sampled_2, v_gen_greedy], dim=0)
    results_dir = os.path.join(config['train_params']['task_name'], "cftn_v2_results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"eval_{tag}_comparison.png")
    save_image(vis_grid, save_path, nrow=3) 

    # --- Part 4: TEXT CAPTIONING EVALUATION (New) ---
    # We will take the validation images (v_imgs_gpu) and ask the model to describe them.
    # Since transformers are autoregressive for text, we generate word by word.
    
    print(f"\n📝 Evaluating Captioning on {len(v_imgs_gpu)} images...")
    
    # Get image indices from VQ-VAE
    with torch.no_grad():
        v_img_indices = vq_model.get_indices(v_imgs_gpu) # Shape: [3, 256] (or block_size)
    
    # Start tokens: [CLS] for BERT-based tokenizers (id 101)
    # We create a tensor of shape [Batch, 1] containing the start token
    start_token_id = tokenizer.cls_token_id
    gen_text_ids = torch.tensor([[start_token_id]] * v_imgs_gpu.size(0), device=device)
    
    # Loop to generate text (max length 30 tokens for speed)
    max_gen_len = 30 
    for _ in range(max_gen_len):
        # Predict next token based on Image + Current Text Sequence
        txt_logits, _ = model(img_indices=v_img_indices, text_ids=gen_text_ids)
        
        # Look at the logits of the LAST token in the sequence
        next_token_logits = txt_logits[:, -1, :] 
        
        # Greedy decoding (pick best word)
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
        
        # Append to sequence
        gen_text_ids = torch.cat((gen_text_ids, next_token), dim=1)
        
        # Stop early if all generated [SEP] (optional, skipped here for simplicity)

    # Decode tokens to string
    generated_captions = tokenizer.batch_decode(gen_text_ids, skip_special_tokens=True)
    gt_captions_text = v_captions[:3] # Original ground truth strings

    # Print and Prepare WandB Table
    text_table_data = []
    print("-" * 60)
    for i in range(len(generated_captions)):
        print(f"🖼️ Img {i} GT:   {gt_captions_text[i]}")
        print(f"       Pred: {generated_captions[i]}")
        text_table_data.append([tag, gt_captions_text[i], generated_captions[i]])
    print("-" * 60)
    print(f"📊 Eval {tag} | PSNR: {avg_psnr:.2f} | Image Saved: {save_path}")

    if use_wandb:
        wandb.log({
            "brain/eval_psnr": avg_psnr,
            "brain/generations_comparison": wandb.Image(save_path, caption=f"R1: Greedy/T0.8/T1.0"),
            # Log the captions as a table
            "brain/caption_samples": wandb.Table(columns=["Step", "Ground Truth", "Generated"], data=text_table_data)
        })
    
    model.train()

def train(args):
    # 1. Load Config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Setup Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    full_dataset = NarutoDataset(image_size=config['model_params']['image_size'])
    dataset_size = len(full_dataset)
    val_size = 16 
    train_size = dataset_size - val_size
    
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    loader = DataLoader(train_ds, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    
    # 3. Load VQ-VAE
    vqvae = get_vqvae(config)
    vqvae_ckpt = os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name'])
    if os.path.exists(vqvae_ckpt):
        print(f"Loading VQ-VAE from {vqvae_ckpt}")
        vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device, weights_only=True))
    vqvae.eval()
    for p in vqvae.parameters(): p.requires_grad = False

    # 4. Setup Brain Model
    img_size = config['model_params']['image_size']
    downsample = 2 ** config['model_params']['num_downsampling_layers']
    block_size = (img_size // downsample) ** 2
    
    config['cftn_v2_params'].update({
        'block_size': block_size,
        'vocab_size': config['model_params']['num_embeddings'],
        'text_vocab_size': tokenizer.vocab_size
    })
    
    model = BiHemisphericBrain(config['cftn_v2_params']).to(device)
    
    gate_params = [p for n, p in model.named_parameters() if 'gate' in n]
    other_params = [p for n, p in model.named_parameters() if 'gate' not in n]
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': config['cftn_v2_params']['lr_text']}, 
        {'params': gate_params, 'lr': config['cftn_v2_params']['lr_gate']}
    ])
    
    # Losses
    criterion_img = nn.CrossEntropyLoss()
    criterion_text = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
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
        print("🚀 STAGE 2A: Bidirectional Training...")
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
                
                # 1. Text -> Image
                _, vis_logits = model(img_indices=None, text_ids=text_ids)
                loss_g = criterion_img(vis_logits.reshape(-1, config['model_params']['num_embeddings']), img_indices.reshape(-1))
                
                # 2. Image -> Text
                txt_logits, _ = model(img_indices=img_indices, text_ids=text_ids[:, :-1])
                loss_c = criterion_text(txt_logits.reshape(-1, tokenizer.vocab_size), text_ids[:, 1:].reshape(-1))
                
                loss = loss_g + loss_c
                loss.backward()
                optimizer.step()
                
                preds = torch.argmax(vis_logits, dim=-1)
                acc = (preds == img_indices).float().mean().item() * 100
                loss_gen.append(loss_g.item())
                loss_cap.append(loss_c.item())
                acc_gen.append(acc)
                
                pbar.set_description(f"Ep {epoch} | GenL: {np.mean(loss_gen):.3f} | CapL: {np.mean(loss_cap):.3f} | Acc: {np.mean(acc_gen):.1f}%")

            if args.use_wandb:
                wandb.log({
                    "brain/gen_loss": np.mean(loss_gen), 
                    "brain/cap_loss": np.mean(loss_cap), 
                    "brain/vis_accuracy": np.mean(acc_gen),
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
                loss = criterion_img(vis_logits.reshape(-1, config['model_params']['num_embeddings']), img_indices.reshape(-1))
                
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