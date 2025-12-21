## Project Overview

This project is a PyTorch implementation of a Vector-Quantized Variational Autoencoder (VQ-VAE) for image generation. It has evolved from a basic implementation to a **VQ-VAE v2** architecture, supporting residual stacks and EMA-based quantization. It is currently configured to work with the Hugging Face dataset `lambdalabs/naruto-blip-captions`.

The project structure is:

-   `config/`: YAML files for model and training hyperparameters (currently focusing on `vqvae_naruto.yaml`).
-   `dataset/`: Dataset loading logic, featuring `NarutoDataset` for Hugging Face integration.
-   `model/`: Core VQ-VAE v2 components (`vqvae_v2.py`, `quantizer_v2.py`). Legacy v1 files (`encoder.py`, `decoder.py`) may remain.
-   `tools/`: Scripts for training (`train_vqvae.py`), inference (`infer_vqvae.py`), and downstream LSTM training for generation.

## Current Architecture (VQ-VAE v2)

-   **Encoder**: Initial convolution followed by downsampling layers and a `ResidualStack`.
-   **Quantizer**: `VectorQuantizerEMA` which uses Exponential Moving Averages for codebook updates, providing more stable training than standard VQ.
-   **Decoder**: A `ResidualStack` followed by upsampling transpose convolutions to return to the original image dimensions.
-   **Loss Function**: Combined MSE reconstruction loss and a commitment loss (weighted by `commitment_cost`).

## Building and Running

### 1. Setup Environment

```bash
pip install -r requirements.txt
wandb login
```

### 2. Training the VQ-VAE

Training uses `wandb` for logging. Ensure your configuration in `config/vqvae_naruto.yaml` is correct before starting.

```bash
python -m tools.train_vqvae --config config/vqvae_naruto.yaml
```

### 3. Monitoring

-   **WandB**: View real-time loss curves, perplexity, and reconstruction grids.
-   **Local Results**: Reconstruction grids are saved in the `task_name/results` directory.

## Development Conventions

-   **EMA Quantization**: Prefer using the EMA quantizer for better codebook utilization.
-   **Residual Blocks**: Residual connections are key to training deeper VQ-VAE models on complex datasets.
-   **Dataset Agnostic**: Tools are designed to be dataset-agnostic by utilizing the `dataset` key in configuration files.
