## Project Overview

This project is a PyTorch implementation of a Vector-Quantized Variational Autoencoder (VQ-VAE) for image generation. It has evolved from a basic implementation to a **VQ-VAE v2** architecture, supporting residual stacks and EMA-based quantization. It is currently configured to work with the Hugging Face dataset `lambdalabs/naruto-blip-captions`.

The project also includes multiple generative models that operate on the learned latent codes:
1.  **LSTM**: A standard autoregressive sequence model.
2.  **Transformer**: A GPT-style autoregressive model for global coherence.
3.  **CFTN (MaskGIT)**: A bidirectional transformer that uses masked image token modeling for faster, non-autoregressive parallel decoding.

The project structure is:

-   `config/`: YAML files for model and training hyperparameters.
-   `dataset/`: Dataset loading logic, featuring `NarutoDataset`.
-   `model/`: Core model components (`vqvae_v2.py`, `transformer.py`, `cftn.py`).
-   `tools/`: Scripts for training and inference for all architectures.

## Current Architectures

### 1. VQ-VAE v2
-   **Encoder**: Downsampling layers with a `ResidualStack`.
-   **Quantizer**: `VectorQuantizerEMA` for stable codebook updates.
-   **Decoder**: Upsampling layers with a `ResidualStack`.

### 2. CFTN (Conditional Flow Transformer Network)
-   Inspired by **MaskGIT**.
-   Uses a bidirectional transformer with cross-attention for text conditioning.
-   **Training**: Predicts randomly masked image tokens.
-   **Inference**: Uses iterative parallel decoding, filling in the most confident tokens first.

## Building and Running

### 1. Setup Environment

```bash
pip install -r requirements.txt
wandb login
```

### 2. Training the VQ-VAE

```bash
python -m tools.train_vqvae --config config/vqvae_naruto.yaml
```

### 3. Training the Generative Models

**LSTM:**
```bash
python -m tools.train_lstm --config config/vqvae_naruto.yaml
```

**Transformer:**
```bash
python -m tools.train_transformer --config config/vqvae_naruto.yaml
```

**CFTN (MaskGIT):**
```bash
python -m tools.train_cftn
```

### 4. Image Generation

**Transformer:**
```bash
python -m tools.generate_transformer --config config/vqvae_naruto.yaml --num_samples 16 --temp 0.8
```

**CFTN:**
```bash
python -m tools.generate_cftn --prompt "Naruto in the hidden leaf village" --steps 12 --num_samples 4
```

## Development Conventions

-   **WandB Integration**: All training and generation scripts log metrics and samples to `wandb`.
-   **Checkpointing**: Models save the "best" version based on training progress to the `task_name` directory.
-   **Reproducibility**: Use the `seed` parameter in configuration for consistent results.