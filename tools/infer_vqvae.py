import yaml
import argparse
import torch
import os
from tqdm import tqdm
import torchvision
from model.vqvae import get_model
from torch.utils.data.dataloader import DataLoader
from dataset.mnist_dataset import MnistDataset
from dataset.naruto_dataset import NarutoDataset
from torchvision.utils import make_grid
from einops import rearrange
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataset(config, split):
    if config['train_params']['dataset'] == 'mnist':
        dataset = MnistDataset(split, config['train_params']['train_path'], im_channels=config['model_params']['in_channels'])
    elif config['train_params']['dataset'] == 'naruto':
        dataset = NarutoDataset(split=split)
    else:
        raise ValueError("Unknown dataset")
    return dataset

def reconstruct(config, model, dataset, num_images=100):
    r"""
    Randomly sample points from the dataset and visualize image and its reconstruction
    :param config: Config file used to create the model
    :param model: Trained model
    :param dataset: Dataset (not the data loader)
    :param num_images: NUmber of images to visualize
    :return:
    """
    print('Generating reconstructions')
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])
    if not os.path.exists(
            os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir'])):
        os.mkdir(os.path.join(config['train_params']['task_name'], config['train_params']['output_train_dir']))
    
    idxs = torch.randint(0, len(dataset) - 1, (num_images,))
    ims = torch.cat([dataset[idx][0][None, :] for idx in idxs]).float()
    ims = ims.to(device)
    model_output = model(ims)
    output = model_output['generated_image']
    
    # Dataset generates -1 to 1 we convert it to 0-1
    ims = (ims + 1) / 2
    
    generated_im = (output + 1) / 2
    out = torch.hstack([ims, generated_im])
    output = rearrange(out, 'b (c d) h w -> b (d) h (c w)', c=2, d=config['model_params']['in_channels'])
    
    grid = make_grid(output.detach().cpu(), nrow=10)
    
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
        model_output = model(im)
        quant_indices = model_output['quantized_indices']
        save_encodings = quant_indices if save_encodings is None else torch.cat([save_encodings, quant_indices], dim=0)
    pickle.dump(save_encodings, open(os.path.join(config['train_params']['task_name'],
                                                  config['train_params']['output_train_dir'],
                                                  'encodings.pkl'), 'wb'))
    print('Done saving encoder outputs for lstm for training')


def inference(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(os.path.join(config['train_params']['task_name'],
                                                  config['train_params']['ckpt_name']), map_location='cpu'))
    model.to(device)
    model.eval()
    
    train_dataset = get_dataset(config, 'train')
    test_dataset = get_dataset(config, 'train') # Using train split for naruto as there is no test split.
    
    train_loader = DataLoader(train_dataset, batch_size=config['train_params']['batch_size'], shuffle=False, num_workers=4)
    
    with torch.no_grad():
        # Generate Reconstructions
        reconstruct(config, model, test_dataset)
        # Save Encoder Outputs for training lstm
        save_encodings(config, model, train_loader)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vqvae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/vqvae_naruto.yaml', type=str)
    args = parser.parse_args()
    inference(args)