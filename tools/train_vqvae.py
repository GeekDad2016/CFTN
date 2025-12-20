import yaml
import argparse
import torch
import cv2
import random
import os
import shutil
import torchvision
import numpy as np
from tqdm import tqdm
from model.vqvae import get_model
from torch.utils.data.dataloader import DataLoader
from dataset.naruto_dataset import NarutoDataset
from torch.optim import Adam
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataset(config):
    if config['train_params']['dataset'] == 'mnist':
        raise ValueError("MNIST dataset is no longer supported.")
    elif config['train_params']['dataset'] == 'naruto':
        dataset = NarutoDataset()
    else:
        raise ValueError("Unknown dataset")
    return dataset

def train_for_one_epoch(epoch_idx, model, data_loader, optimizer, criterion, config, val_images):
    r"""
    Method to run the training for one epoch.
    :param epoch_idx: iteration number of current epoch
    :param model: VQVAE model
    :param data_loader: Data loader for the dataset
    :param optimizer: optimizer to be used taken from config
    :param criterion: For computing the loss
    :param config: configuration for the current run
    :param val_images: A fixed set of validation images for logging
    :return:
    """
    recon_losses = []
    codebook_losses = []
    commitment_losses = []
    losses = []
    for i, (im, _) in enumerate(tqdm(data_loader)):
        im = im.float().to(device)
        optimizer.zero_grad()
        model_output = model(im)
        output = model_output['generated_image']
        quantize_losses = model_output['quantized_losses']

        recon_loss = criterion(output, im)
        loss = (config['train_params']['reconstruction_loss_weight'] * recon_loss +
                config['train_params']['codebook_loss_weight'] * quantize_losses['codebook_loss'] +
                config['train_params']['commitment_loss_weight'] * quantize_losses['commitment_loss'])
        
        recon_losses.append(recon_loss.item())
        codebook_losses.append(quantize_losses['codebook_loss'].item())
        commitment_losses.append(quantize_losses['commitment_loss'].item())
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    wandb.log({
        "epoch": epoch_idx,
        "recon_loss": np.mean(recon_losses),
        "codebook_loss": np.mean(codebook_losses),
        "commitment_loss": np.mean(commitment_losses),
        "total_loss": np.mean(losses)
    })

    print(f'Finished epoch: {epoch_idx + 1} | Recon Loss : {np.mean(recon_losses):.4f} | '
          f'Codebook Loss : {np.mean(codebook_losses):.4f} | Commitment Loss : {np.mean(commitment_losses):.4f}')
    
    if config['train_params']['save_training_image']:
        output_dir = os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir'], 'results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create a grid of images
        input_grid = make_grid((im.detach() + 1) / 2)
        output_grid = make_grid((output.detach() + 1) / 2)
        
        # Convert to numpy and save
        input_numpy = (255 * input_grid).cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        output_numpy = (255 * output_grid).cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f'input_epoch_{epoch_idx}.jpeg'), cv2.cvtColor(input_numpy, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f'output_epoch_{epoch_idx}.jpeg'), cv2.cvtColor(output_numpy, cv2.COLOR_RGB2BGR))
    
    with torch.no_grad():
        model.eval()
        reconstructions = model(val_images)['generated_image']
        grid = make_grid(torch.cat([val_images, reconstructions], dim=0), nrow=val_images.size(0))
        wandb.log({"reconstructions": wandb.Image(grid)})
        model.train()

    return np.mean(losses)

def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    wandb.init(project="vqvae-naruto", config=config)
    
    print(config)
    print(f"Using device: {device}")
    
    seed = config['train_params']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    model = get_model(config).to(device)
    dataset = get_dataset(config)
    data_loader = DataLoader(dataset, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    
    val_images, _ = next(iter(DataLoader(dataset, batch_size=16, shuffle=True)))
    val_images = val_images.to(device)

    num_epochs = config['train_params']['epochs']
    optimizer = Adam(model.parameters(), lr=config['train_params']['lr'])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)
    criterion = {'l1': torch.nn.L1Loss(), 'l2': torch.nn.MSELoss()}.get(config['train_params']['crit'])
    
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])
    if not os.path.exists(os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir'])):
        os.mkdir(os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir']))
    
    if os.path.exists(os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name'])):
        print('Loading checkpoint')
        model.load_state_dict(torch.load(os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name']), map_location=device))
    
    best_loss = np.inf
    
    for epoch_idx in range(num_epochs):
        mean_loss = train_for_one_epoch(epoch_idx, model, data_loader, optimizer, criterion, config, val_images)
        scheduler.step(mean_loss)
        if mean_loss < best_loss:
            print(f'Improved Loss to {mean_loss:.4f} .... Saving Model')
            torch.save(model.state_dict(), os.path.join(config['train_params']['task_name'], config['train_params']['ckpt_name']))
            best_loss = mean_loss
        else:
            print('No Loss Improvement')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path', default='config/vqvae_naruto.yaml', type=str)
    args = parser.parse_args()
    train(args)