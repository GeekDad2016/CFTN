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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VQVAELSTM(nn.Module):
    r"""
    Very Simple 2 layer LSTM with an fc layer on last steps hidden dimension
    """
    def __init__(self, input_size, hidden_size, codebook_size):
        super(VQVAELSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
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
        self.codebook_size = config['model_params']['codebook_size']
        
        # Codebook tokens will be 0 to codebook_size-1
        self.start_token = self.codebook_size
        self.pad_token = self.codebook_size+1
        # Fix context size
        self.context_size = 32
        self.sents = self.load_sents(config)
    
    def load_sents(self, config):
        encoding_path = os.path.join(config['train_params']['task_name'],
                                           config['train_params']['output_train_dir'],
                                           'encodings.pkl')
        assert os.path.exists(encoding_path), ("No encodings generated for lstm. "
                                               "Run save_encodings method in inference script")
        
        encodings = pickle.load(open(encoding_path, 'rb'))
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
            
            # Make sure all encodings start with start token
            enc = torch.cat([torch.ones((1)).to(device) * self.start_token, enc.to(device)])
            
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

def train_lstm(args):
    ############ Read the config #############
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    #########################################
    
    ############## Create dataset ###########
    dataset = VQVAESeqDataset(config)
    data_loader = DataLoader(dataset, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    #########################################
    
    ############## Create LSTM ###########
    model = VQVAELSTM(input_size=config['lstm_params']['embedding_dim'],
                      hidden_size=config['lstm_params']['hidden_size'],
                      codebook_size=config['model_params']['codebook_size']).to(device)
    model.to(device)
    model.train()
    
    ############## Training Params ###########
    num_epochs = config['lstm_params']['epochs']
    optimizer = Adam(model.parameters(), lr=1E-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
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
        print('Epoch {} : {}'.format(epoch, np.mean(losses)))
        print('=' * 50)
        torch.save(model.state_dict(), os.path.join(config['train_params']['task_name'],
                                                    'best_lstm.pth'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for lstm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/vqvae_naruto.yaml', type=str)
    args = parser.parse_args()
    train_lstm(args)
    
