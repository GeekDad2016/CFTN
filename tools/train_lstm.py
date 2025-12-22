import torch
import torch.nn as nn
import os
import cv2
import glob
import torch
import yaml
import argparse
import random
import numpy as np
from torch.optim import Adam
import pickle
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from model.vqvae_v2 import VQVAEv2
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VQVAELSTM(nn.Module):
    r"""
    Simple 4 layer LSTM with an fc layer on last steps hidden dimension
    """
    def __init__(self, input_size, hidden_size, codebook_size):
        super(VQVAELSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=4, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size // 4),
                                nn.ReLU(),
                                nn.Linear(hidden_size // 4, codebook_size))
        # Add pad and start token to embedding size
        self.word_embedding = nn.Embedding(codebook_size+2, input_size)
    
    def forward(self, x):
        x = self.word_embedding(x)
        output, _ = self.rnn(x)
        output = output[:, -1, :]
        return self.fc(output)


class VQVAESeqDataset(Dataset):
    r"""
    Dataset for training of LSTM. Assumes the encodings are already generated
    by running vqvae inference
    """
    def __init__(self, config):
        self.codebook_size = config['model_params']['num_embeddings']
        
        # Codebook tokens will be 0 to codebook_size-1
        self.start_token = self.codebook_size
        self.pad_token = self.codebook_size+1
        # Fix context size
        self.context_size = 128
        self.sents = self.load_sents(config)
    
    def load_sents(self, config):
        encoding_path = os.path.join(config['train_params']['task_name'],
                                           config['train_params']['output_train_dir'],
                                           'encodings.pkl')
        assert os.path.exists(encoding_path), ("No encodings generated for lstm. "
                                               "Run save_encodings method in inference script")
        
        encodings = pickle.load(open(encoding_path, 'rb'))
        # Store latent shape for generation
        self.latent_shape = (encodings.shape[1], encodings.shape[2])
        
        encodings = encodings.reshape(encodings.size(0), -1)
        num_encodings = encodings.size(0)
        padded_sents = []
        
        for encoding_idx in tqdm(range(num_encodings)):
            # Use only 10% encodings.
            # Uncomment this for getting some kind of output quickly validate working
            if random.random() > 0.1:
                continue
            enc = encodings[encoding_idx]
            encoding_length = enc.shape[-1]
            
            # Make sure all encodings start with start token. Keep on CPU for DataLoader compatibility.
            enc = torch.cat([torch.ones((1)) * self.start_token, enc.cpu()])
            
            # Create batches of context sized inputs(if possible) and target
            sents = [(enc[:i], enc[i]) if i < self.context_size else (enc[i - self.context_size:i], enc[i])
                   for i in range(1, encoding_length+1)]
            
            for context, target in sents:
                # Pad token if context not enough
                if len(context) < self.context_size:
                    context = torch.nn.functional.pad(context, (0, self.context_size-len(context)), "constant", self.pad_token)
                padded_sents.append((context, target))
        return padded_sents
    
    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, index):
        context, target = self.sents[index]
        return context, target

@torch.no_grad()
def generate_samples(model, vqvae_model, config, latent_shape, epoch):
    print(f"Generating samples at epoch {epoch}...")
    model.eval()
    vqvae_model.eval()
    
    num_embeddings = config['model_params']['num_embeddings']
    num_samples = 16
    context_size = 32
    encodings_length = latent_shape[0] * latent_shape[1]
    
    generated_indices = []
    for _ in range(num_samples):
        ctx = torch.ones((1)).to(device) * num_embeddings
        for _ in range(encodings_length):
            padded_ctx = ctx
            if len(ctx) < context_size:
                padded_ctx = torch.nn.functional.pad(padded_ctx, (0, context_size - len(ctx)), "constant", num_embeddings + 1)
            else:
                padded_ctx = ctx[-context_size:]
            
            out = model(padded_ctx[None, :].long())
            probs = torch.nn.functional.softmax(out, dim=-1)
            pred = torch.multinomial(probs[0], num_samples=1)
            ctx = torch.cat([ctx, pred])
        generated_indices.append(ctx[1:][None, :])
    
    generated_indices = torch.cat(generated_indices, dim=0)
    quantized_indices = generated_indices.reshape((num_samples, latent_shape[0], latent_shape[1])).long()
    
    embeddings = vqvae_model.vq.embedding(quantized_indices)
    embeddings = embeddings.permute(0, 3, 1, 2).contiguous()
    
    recon = vqvae_model.decoder(embeddings)
    recon = (recon * 0.5 + 0.5).clamp(0, 1)
    
    grid = make_grid(recon.cpu(), nrow=4)
    wandb.log({f"samples/gen": wandb.Image(grid), "epoch": epoch})
    model.train()

def train_lstm(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    wandb.init(project="vqvae-naruto-lstm", config=config)
    print(config)
    
    dataset = VQVAESeqDataset(config)
    data_loader = DataLoader(dataset, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    
    model = VQVAELSTM(input_size=config['lstm_params']['embedding_dim'],
                      hidden_size=config['lstm_params']['hidden_size'],
                      codebook_size=config['model_params']['num_embeddings']).to(device)
    model.to(device)
    
    # Load VQVAE for evaluation
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
    if os.path.exists(vqvae_ckpt):
        vqvae_model.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    vqvae_model.eval()

    num_epochs = config['lstm_params']['epochs']
    optimizer = Adam(model.parameters(), lr=1E-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Load checkpoint if found
    start_epoch = 0
    checkpoint_path = os.path.join(config['train_params']['task_name'], 'best_lstm.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading LSTM checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            # Backwards compatibility for state-dict only checkpoints
            model.load_state_dict(checkpoint)
    
    model.train()
    
    for epoch in range(start_epoch, num_epochs):
        losses = []
        for sent, target in tqdm(data_loader):
            sent = sent.to(device).long()
            target = target.to(device).long()
            optimizer.zero_grad()
            pred = model(sent)
            loss = torch.mean(criterion(pred, target))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        mean_loss = np.mean(losses)
        print(f'Epoch {epoch} : {mean_loss}')
        wandb.log({"epoch": epoch, "loss": mean_loss})
        
        if epoch % 20 == 0:
            generate_samples(model, vqvae_model, config, dataset.latent_shape, epoch)

        print('=' * 50)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(config['train_params']['task_name'], 'best_lstm.pth'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for lstm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/vqvae_naruto.yaml', type=str)
    args = parser.parse_args()
    train_lstm(args)
