## Project Overview

This project is a PyTorch implementation of a Vector-Quantized Variational Autoencoder (VQ-VAE) for image generation. It has evolved from a basic implementation to a **VQ-VAE v2** architecture, supporting residual stacks and EMA-based quantization. It is currently configured to work with the Hugging Face dataset `lambdalabs/naruto-blip-captions`.

The project also includes multiple generative models that operate on the learned latent codes:
1.  **LSTM**: A standard autoregressive sequence model.
2.  **Transformer**: A GPT-style autoregressive model for global coherence.
3.  **CFTN (MaskGIT)**: A bidirectional transformer using masked image token modeling.
4.  **CFTN v2 (BiHemisphericBrain)**: A dual-tower transformer architecture with gated bidirectional bridges, using a two-stage curriculum training process (Bidirectional -> Gen-Only). Designed for **One-Shot Generation** with extreme inference speed.

The project structure is:

-   `config/`: YAML files for model and training hyperparameters.
-   `dataset/`: Dataset loading logic, featuring `NarutoDataset`.
-   `model/`: Core model components (`vqvae_v2.py`, `cftn.py`, `cftn_v2.py`).
-   `tools/`: Scripts for training and inference for all architectures.

## Current Architectures

### 1. VQ-VAE v2
-   **Encoder**: Downsampling layers with a `ResidualStack`.
-   **Quantizer**: `VectorQuantizerEMA` for stable codebook updates.
-   **Decoder**: Upsampling layers with a `ResidualStack`.

### 2. CFTN v2 (BiHemisphericBrain)
-   Inspired by brain connectivity.
-   **Architecture**: Dual transformers (Text Hemisphere + Image Hemisphere).
-   **Bridges**: Gated cross-attention layers (Callosal layers) for bidirectional information exchange.
-   **Performance**: Achieves extreme inference speed via **One-Shot Generation**, mapping semantic concepts to visual tokens in a single pass.
-   **Training**: Two-stage curriculum training.
    -   Stage 1: Bidirectional (Gen + Captioning).
    -   Stage 2: Fine-tuning for One-Shot Generation.

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

**CFTN v2 (Recommended):**
```bash
python -m tools.train_cftn_v2 --config config/vqvae_naruto.yaml --use_wandb
```

**Original CFTN (MaskGIT):**
```bash
python -m tools.train_cftn --config config/vqvae_naruto.yaml
```

**Transformer/LSTM:**
```bash
python -m tools.train_transformer --config config/vqvae_naruto.yaml
python -m tools.train_lstm --config config/vqvae_naruto.yaml
```

### 4. Image Generation

**CFTN v2:**
Generation is currently integrated into the training loop for validation. It performs **One-Shot Generation** from text.

**Original CFTN:**
```bash
python -m tools.generate_cftn --prompt "Naruto in the hidden leaf village" --steps 12 --num_samples 4
```

## Development Conventions

-   **WandB Integration**: All training and generation scripts log metrics and samples to `wandb`.
-   **Checkpointing**: Models save state dictionaries and training metadata to the task directory.
-   **Resumability**: Scripts support resuming from checkpoints automatically.
-   **One-Shot Focus**: CFTN v2 prioritizes speed and semantic coherence via its dual-tower design.