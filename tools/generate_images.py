import argparse
import yaml
import os
import torch
import pickle
from tqdm import tqdm
import torchvision
from model.vqvae_v2 import VQVAEv2
from torchvision.utils import make_grid
from tools.train_lstm import VQVAELSTM
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate(args):
    r"""
    Method for generating images after training vqvae and lstm
    1. Create config
    2. Create and load vqvae model
    3. Create and load LSTM model
    4. Generate 100 encoder outputs from trained LSTM
    5. Pass them to the trained vqvae decoder
    6. Save the generated image
    :param args:
    :return:
    """
    
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            
    wandb.init(project="vqvae-naruto-generation", config=config)
    print(config)
    
    # Initialize VQVAEv2 correctly
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
    
    assert os.path.exists(os.path.join(config['train_params']['task_name'],
                                                  config['train_params']['ckpt_name'])), "Train the vqvae model first"
    vqvae_model.load_state_dict(torch.load(os.path.join(config['train_params']['task_name'],
                                                  config['train_params']['ckpt_name']), map_location=device))
    vqvae_model.eval()
    
    # Initialize VQVAELSTM correctly
    model = VQVAELSTM(input_size=config['lstm_params']['embedding_dim'],
                      hidden_size=config['lstm_params']['hidden_size'],
                      codebook_size=config['model_params']['num_embeddings']).to(device)
    model.to(device)
    assert os.path.exists(os.path.join(config['train_params']['task_name'],
                                                    'best_lstm.pth')), "Train the lstm first"
    model.load_state_dict(torch.load(os.path.join(config['train_params']['task_name'],
                                                    'best_lstm.pth'), map_location=device))
    model.eval()
    
    generated_quantized_indices = []
    encodings_path = os.path.join(config['train_params']['task_name'],
                                  config['train_params']['output_train_dir'],
                                  'encodings.pkl')
    assert os.path.exists(encodings_path), "Generate encodings first"
    
    encodings = pickle.load(open(encodings_path, 'rb'))
    # Calculate length from encodings. Shape is likely [N, H, W]
    encodings_length = encodings.shape[1] * encodings.shape[2]
    
    context_size = 32
    num_samples = 64 # Match batch size for a nice grid
    print('Generating Samples')
    for _ in tqdm(range(num_samples)):
        # Initialize with start token
        ctx = torch.ones((1)).to(device) * (config['model_params']['num_embeddings'])
        
        for i in range(encodings_length):
            padded_ctx = ctx
            if len(ctx) < context_size:
                # Pad context with pad token
                padded_ctx = torch.nn.functional.pad(padded_ctx, (0, context_size - len(ctx)), "constant",
                                                  config['model_params']['num_embeddings']+1)
            elif len(ctx) > context_size:
                # Take only the last 'context_size' tokens
                padded_ctx = ctx[-context_size:]
                
            out = model(padded_ctx[None, :].long().to(device))
            probs = torch.nn.functional.softmax(out, dim=-1)
            pred = torch.multinomial(probs[0], num_samples=1)
            ctx = torch.cat([ctx, pred])
        # Skip the start token
        generated_quantized_indices.append(ctx[1:][None, :])
        
    generated_quantized_indices = torch.cat(generated_quantized_indices, dim=0)
    # Reconstruct shape (assuming square latent grid)
    h_latent = int(encodings.shape[1])
    w_latent = int(encodings.shape[2])
    quantized_indices = generated_quantized_indices.reshape((num_samples, h_latent, w_latent)).long()
    
    # Map indices to embeddings
    # quantized_indices is (B, H, W)
    # vqvae_model.vq.embedding.weight is (num_embeddings, embedding_dim)
    
    # Use embedding lookup
    embeddings = vqvae_model.vq.embedding(quantized_indices) # (B, H, W, embedding_dim)
    embeddings = embeddings.permute(0, 3, 1, 2).contiguous() # (B, embedding_dim, H, W)
    
    with torch.no_grad():
        output = vqvae_model.decoder(embeddings)
        output = (output * 0.5 + 0.5).clamp(0, 1)
    
    grid = make_grid(output.detach().cpu(), nrow=8)
    
    wandb.log({"generated_images": wandb.Image(grid)})
    
    img = torchvision.transforms.ToPILImage()(grid)
    results_path = os.path.join(config['train_params']['task_name'],
                                config['train_params']['output_train_dir'],
                                'generation_results.png')
    img.save(results_path)
    print(f"Saved generation results to {results_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for LSTM generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/vqvae_naruto.yaml', type=str)
    args = parser.parse_args()
    generate(args)