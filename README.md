# Naruto Image Generation: VQ-VAE v2 + CFTN (MaskGIT)

This repository implements a state-of-the-art image generation pipeline using a **VQ-VAE v2** for visual tokenization and a **CFTN (Conditional Flow Transformer Network)** for text-to-image generation.

## 🚀 Overview

The project has evolved from simple sequence modeling (LSTM) to a bidirectional, non-autoregressive transformer approach inspired by **MaskGIT**.

### Key Components:
1.  **VQ-VAE v2**: A robust visual tokenizer that maps $128 \times 128$ images into a discrete $32 \times 32$ grid of latent codes.
2.  **CFTN (MaskGIT style)**: A Bidirectional Transformer with Cross-Attention. It doesn't generate images one-by-one; instead, it learns to "fill in the blanks" of a masked image grid, conditioned on text prompts.

---

## 🛠 Architecture & Mechanism

### 1. VQ-VAE v2 (The Tokenizer)
-   **Encoder/Decoder**: Utilizes deep Residual Blocks to preserve high-frequency details.
-   **Quantizer**: `VectorQuantizerEMA` for stable training and high codebook utilization (4096 embeddings).
-   **Objective**: Minimize Reconstruction Loss (MSE/L1) + Commitment Loss.

### 2. CFTN (The Generator)
-   **Training Objective**: **Masked Image Modeling**. During training, we randomly mask a percentage of image tokens (following a cosine schedule) and task the transformer with predicting the original tokens based on the surrounding context and text descriptions.
-   **Text Conditioning**: Uses a BERT tokenizer and cross-attention layers to inject text features into every block of the transformer.
-   **Inference (Parallel Decoding)**: Unlike LSTMs/GPTs that take 1024 steps to generate an image, CFTN uses **Iterative Parallel Decoding**. It can generate high-quality images in just **8-12 steps** by filling in the most confident tokens first.

---

## 🏃 Usage

### 1. Setup
```bash
pip install -r requirements.txt
wandb login
```

### 2. Step 1: Train VQ-VAE
First, train the model to understand the visual "vocabulary" of Naruto images.
```bash
python -m tools.train_vqvae --config config/vqvae_naruto.yaml
```

### 3. Step 2: Generate Encodings
Once VQ-VAE is trained, convert the entire image dataset into discrete tokens.
```bash
python -m tools.infer_vqvae --config config/vqvae_naruto.yaml
```

### 4. Step 3: Train CFTN (MaskGIT)
Train the transformer to predict masked patches based on text captions.
```bash
python -m tools.train_cftn
```

### 5. Step 4: Generate New Images
Generate images from text prompts using the fast parallel decoding mechanism.
```bash
python -m tools.generate_cftn --prompt "Naruto in the hidden leaf village, high quality digital art" --steps 12 --num_samples 4
```

---

## 📊 Monitoring
All training and generation progress is logged to **Weights & Biases (WandB)**.
-   **VQ-VAE**: Tracks reconstruction grids and perplexity.
-   **CFTN**: Tracks masking loss and provides "in-painting" samples during training.

## 💾 Model Checkpoints
Checkpoints are automatically saved in the `task_name` directory specified in your config. 
-   The scripts support **automatic resuming**: if a training run is interrupted, simply run the command again to pick up from the last saved epoch.