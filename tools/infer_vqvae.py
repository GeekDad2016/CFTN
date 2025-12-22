import yaml
import argparse
import torch
import os
from tqdm import tqdm
import torchvision
from model.vqvae_v2 import VQVAEv2
from torch.utils.data.dataloader import DataLoader
from dataset.naruto_dataset import NarutoDataset
from torchvision.utils import make_grid
from einops import rearrange
import pickle
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataset(config, split):
    if config['train_params']['dataset'] == 'naruto':
        dataset = NarutoDataset(split=split, image_size=config['model_params']['image_size'])
    else:
        raise ValueError("Unknown dataset")
    return dataset

def reconstruct(config, model, dataset, num_images=100):
    r"""
    Randomly sample points from the dataset and visualize image and its reconstruction
    :param config: Config file used to create the model
    :param model: Trained model
    :param dataset: Dataset (not the data loader)
    :param num_images: Number of images to visualize
    :return:
    """
    print('Generating reconstructions')
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])
    if not os.path.exists(
            os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir'])):
        os.mkdir(os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir']))
    
    idxs = torch.randint(0, len(dataset) - 1, (num_images,))
    ims = torch.cat([dataset[idx.item()][0][None, :] for idx in idxs]).float()
    ims = ims.to(device)
    
    # VQVAEv2 returns vq_loss, x_recon, perplexity, encoding_indices
    _, output, _, _ = model(ims)
    
    # Dataset generates -1 to 1 we convert it to 0-1
    ims = (ims * 0.5 + 0.5).clamp(0, 1)
    generated_im = (output * 0.5 + 0.5).clamp(0, 1)
    
    # Concatenate side-by-side along the width dimension
    out = torch.cat([ims, generated_im], dim=3)
    
    grid = make_grid(out, nrow=5) # 5 pairs per row
    
    wandb.log({"reconstructions": wandb.Image(grid)})
    
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(os.path.join(config['train_params']['task_name'],
                          config['train_params']['output_train_dir'],
                          'reconstruction.png'))


def save_encodings(config, model, data_loader):
    r"""
    Method to save the encoder outputs for training LSTM
    :param config:
    :param model:
    :param data_loader:
    :return:
    """
    save_encodings = None
    print('Saving Encodings for lstm')
    for im, _ in tqdm(data_loader):
        im = im.float().to(device)
        # VQVAEv2 returns vq_loss, x_recon, perplexity, encoding_indices
        _, _, _, quant_indices = model(im)
        save_encodings = quant_indices if save_encodings is None else torch.cat([save_encodings, quant_indices], dim=0)
    
    encoding_path = os.path.join(config['train_params']['task_name'],
                                 config['train_params']['output_train_dir'],
                                 'encodings.pkl')
    # Use torch.save instead of pickle for better compatibility with map_location
    torch.save(save_encodings.cpu(), encoding_path)
    print('Done saving encoder outputs for lstm for training')
    
    artifact = wandb.Artifact('encodings', type='dataset')
    artifact.add_file(encoding_path)
    wandb.log_artifact(artifact)


def inference(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    wandb.init(project="vqvae-naruto-inference", config=config)
    print(config)
    
    model = VQVAEv2(
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
    
    model.load_state_dict(torch.load(os.path.join(config['train_params']['task_name'],
                                                  config['train_params']['ckpt_name']), map_location=device))
    model.eval()
    
    train_dataset = get_dataset(config, 'train')
    
    train_loader = DataLoader(train_dataset, batch_size=config['train_params']['batch_size'], shuffle=False, num_workers=4)
    
    with torch.no_grad():
        # Generate Reconstructions
        reconstruct(config, model, train_dataset)
        # Save Encoder Outputs for training lstm
        save_encodings(config, model, train_loader)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vqvae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/vqvae_naruto.yaml', type=str)
    args = parser.parse_args()
    inference(args)
