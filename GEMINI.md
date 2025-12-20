## Project Overview

This project is a PyTorch implementation of a Vector-Quantized Variational Autoencoder (VQ-VAE) for image generation. Initially designed for the MNIST dataset, it has been extended to support external datasets from Hugging Face, specifically `lambdalabs/naruto-blip-captions`. It includes scripts for training the VQ-VAE, generating image reconstructions, and training a separate LSTM model to generate new images based on the VQ-VAE's learned latent space.

The project is structured as follows:

-   `config/`: Contains YAML configuration files for different VQ-VAE models (e.g., for MNIST and Naruto datasets).
-   `dataset/`: Contains the dataset loading logic, now supporting Hugging Face datasets.
-   `model/`: Contains the core components of the VQ-VAE model, including the encoder, decoder, and quantizer.
-   `tools/`: Contains scripts for training the VQ-VAE, performing inference, and training the generative LSTM.

## Building and Running

### 1. Setup Environment

It is recommended to use a Conda environment.

```bash
conda create -n vqvae python=3.8
conda activate vqvae
pip install -r requirements.txt
```

### 2. Data Preparation

The project can now use Hugging Face datasets directly, such as `lambdalabs/naruto-blip-captions`. For custom datasets, the structure will depend on the dataset loader implementation.

### 3. Running the Models

The project provides several scripts for training and inference. The `--config` argument should point to the appropriate YAML configuration file.

**Simple VQ-VAE:**

To run a minimal VQ-VAE for quick testing and understanding (currently set up for MNIST):

```bash
python run_simple_vqvae.py
```

**Training the VQ-VAE:**

To train the VQ-VAE with a specific configuration (e.g., for Naruto):

```bash
python -m tools.train_vqvae --config config/vqvae_naruto.yaml
```

**Inference with the VQ-VAE:**

To generate reconstructions and save the encoder's output for LSTM training (e.g., for Naruto):

```bash
python -m tools.infer_vqvae --config config/vqvae_naruto.yaml
```

**Training the LSTM:**

To train the LSTM model for generating images (e.g., for Naruto):

```bash
python -m tools.train_lstm --config config/vqvae_naruto.yaml
```

**Generating Images with the LSTM:**

To generate images using the trained LSTM (e.g., for Naruto):

```bash
python -m tools.generate_images --config config/vqvae_naruto.yaml
```

## Development Conventions

-   **Configuration:** The project uses YAML files in the `config/` directory to manage model and training parameters. This allows for easy experimentation with different hyperparameters.
-   **Modularity:** The VQ-VAE model is broken down into three main components: an encoder, a decoder, and a quantizer, each in its own file in the `model/` directory. This promotes code reuse and readability.
-   **Scripts:** Separate scripts are provided for different tasks (training, inference, etc.) in the `tools/` directory. These scripts use `argparse` to accept configuration file paths as command-line arguments.